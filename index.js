require("dotenv").config();
const express = require("express");
const { randomUUID } = require("crypto");

const app = express();
app.use(express.json({ limit: "50mb" }));


const AZURE_ENDPOINT = process.env.AZURE_ENDPOINT;
const AZURE_API_KEY = process.env.AZURE_API_KEY;
const AZURE_MODEL = process.env.AZURE_MODEL || "DeepSeek-V4-Flash";
const PORT = process.env.PORT || 8082;

let clients = [];
let logBuffer = [];
const MAX_LOGS = 100;

function addLog(type, message, data = null) {
  const logEntry = {
    id: randomUUID(),
    timestamp: new Date().toLocaleTimeString(),
    type,
    message,
    data: data ? (typeof data === "object" ? JSON.stringify(sanitizeForLog(data), null, 2) : data) : null
  };
  logBuffer.push(logEntry);
  if (logBuffer.length > MAX_LOGS) logBuffer.shift();
  clients.forEach(c => c.res.write(`data: ${JSON.stringify(logEntry)}\n\n`));
}

function sanitizeForLog(obj) {
  if (!obj || typeof obj !== 'object') return obj;
  return JSON.parse(JSON.stringify(obj, (key, value) => {
    if (typeof value === 'string' && value.length > 200 && (value.includes(';base64,') || key === 'data')) {
      return value.substring(0, 50) + `... [TRUNCATED ${value.length} CHARS]`;
    }
    return value;
  }));
}

function convertAnthropicToOpenAI(body) {
  const messages = [];

  if (body.system) {
    const systemText =
      typeof body.system === "string"
        ? body.system
        : Array.isArray(body.system)
          ? body.system
            .map((b) => (typeof b === "string" ? b : b.text || ""))
            .join("\n")
          : "";
    if (systemText) {
      messages.push({ role: "system", content: systemText });
    }
  }

  for (const msg of body.messages || []) {
    if (msg.role === "user" || msg.role === "assistant") {
      if (typeof msg.content === "string") {
        messages.push({ role: msg.role, content: msg.content });
      } else if (Array.isArray(msg.content)) {
        const toolCalls = [];
        const toolResults = [];
        let textParts = [];
        let contentParts = [];
        let hasImage = false;

        for (const block of msg.content) {
          if (block.type === "text") {
            textParts.push(block.text);
            contentParts.push({ type: "text", text: block.text });
          } else if (block.type === "thinking") {
            continue;
          } else if (block.type === "tool_use") {
            toolCalls.push({
              id: block.id,
              type: "function",
              function: {
                name: block.name,
                arguments:
                  typeof block.input === "string"
                    ? block.input
                    : JSON.stringify(block.input || {}),
              },
            });
          } else if (block.type === "tool_result") {
            let resultContent = "";
            let resultContentParts = [];
            let hasToolImage = false;

            if (typeof block.content === "string") {
              resultContent = block.content;
            } else if (Array.isArray(block.content)) {
              const stringParts = [];
              for (const c of block.content) {
                if (c.type === "text") {
                  stringParts.push(c.text);
                  resultContentParts.push({ type: "text", text: c.text });
                } else if (c.type === "image") {
                  hasToolImage = true;
                  stringParts.push("[image]");
                  if (c.source?.type === "url") {
                    resultContentParts.push({
                      type: "image_url",
                      image_url: {
                        url: c.source.url,
                      },
                    });
                  } else {
                    const mediaType = c.source?.media_type || "image/jpeg";
                    const data = c.source?.data || "";
                    resultContentParts.push({
                      type: "image_url",
                      image_url: {
                        url: `data:${mediaType};base64,${data}`,
                      },
                    });
                  }
                } else {
                  stringParts.push(JSON.stringify(c));
                  resultContentParts.push({ type: "text", text: JSON.stringify(c) });
                }
              }
              resultContent = stringParts.join("\n");
            }
            toolResults.push({
              role: "tool",
              tool_call_id: block.tool_use_id,
              content: hasToolImage ? resultContentParts : resultContent,
            });
          } else if (block.type === "image") {
            hasImage = true;
            textParts.push("[image]");
            if (block.source?.type === "url") {
              contentParts.push({
                type: "image_url",
                image_url: {
                  url: block.source.url,
                },
              });
            } else {
              const mediaType = block.source?.media_type || "image/jpeg";
              const data = block.source?.data || "";
              contentParts.push({
                type: "image_url",
                image_url: {
                  url: `data:${mediaType};base64,${data}`,
                },
              });
            }
          }
        }

        if (toolCalls.length > 0) {
          messages.push({
            role: "assistant",
            content: textParts.join("\n") || null,
            tool_calls: toolCalls,
          });
        } else if (toolResults.length > 0) {
          for (const tr of toolResults) {
            messages.push(tr);
          }
        } else {
          messages.push({
            role: msg.role,
            content: hasImage ? contentParts : (textParts.join("\n") || ""),
          });
        }
      }
    }
  }

  const isDeepSeek = AZURE_MODEL.toLowerCase().includes("deepseek");

  const openaiBody = {
    model: AZURE_MODEL,
    messages,
    stream: !!body.stream,
  };

  if (isDeepSeek) {
    openaiBody.max_tokens = 16384;
    openaiBody.temperature = body.temperature !== undefined ? body.temperature : 0.8;
    openaiBody.top_p = body.top_p !== undefined ? body.top_p : 0.1;
    openaiBody.presence_penalty = 0;
    openaiBody.frequency_penalty = 0;
    openaiBody.reasoning_effort = "high";
  } else {
    openaiBody.max_completion_tokens = 16383;
    if (body.temperature !== undefined) {
      openaiBody.temperature = body.temperature;
    }
    if (body.top_p !== undefined) {
      openaiBody.top_p = body.top_p;
    }
  }

  if (body.stop_sequences) {
    openaiBody.stop = body.stop_sequences;
  }

  if (body.tools && body.tools.length > 0) {
    openaiBody.tools = body.tools
      .filter((t) => t.name && t.input_schema)
      .map((t) => ({
        type: "function",
        function: {
          name: t.name,
          description: t.description || "",
          parameters: t.input_schema || { type: "object", properties: {} },
        },
      }));

    if (body.tool_choice) {
      if (body.tool_choice.type === "any") {
        openaiBody.tool_choice = "required";
      } else if (body.tool_choice.type === "auto") {
        openaiBody.tool_choice = "auto";
      } else if (body.tool_choice.type === "tool" && body.tool_choice.name) {
        openaiBody.tool_choice = {
          type: "function",
          function: { name: body.tool_choice.name },
        };
      }
    }

    if (openaiBody.tools.length === 0) {
      delete openaiBody.tools;
    }
  }

  if (body.stream) {
    openaiBody.stream_options = { include_usage: true };
  }

  return openaiBody;
}

function convertOpenAIToAnthropic(openaiResponse, requestModel) {
  const choice = openaiResponse.choices?.[0];
  if (!choice) {
    return {
      id: `msg_${randomUUID().replace(/-/g, "")}`,
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "" }],
      model: requestModel || AZURE_MODEL,
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: {
        input_tokens: openaiResponse.usage?.prompt_tokens || 0,
        output_tokens: openaiResponse.usage?.completion_tokens || 0,
      },
    };
  }

  const content = [];
  const msg = choice.message;

  if (msg.content) {
    content.push({ type: "text", text: msg.content });
  }

  if (msg.tool_calls && msg.tool_calls.length > 0) {
    for (const tc of msg.tool_calls) {
      let parsedInput = {};
      try {
        parsedInput = JSON.parse(tc.function.arguments || "{}");
      } catch {
        parsedInput = {};
      }
      content.push({
        type: "tool_use",
        id: tc.id || `toolu_${randomUUID().replace(/-/g, "")}`,
        name: tc.function.name,
        input: parsedInput,
      });
    }
  }

  if (content.length === 0) {
    content.push({ type: "text", text: "" });
  }

  let stopReason = "end_turn";
  if (choice.finish_reason === "tool_calls") {
    stopReason = "tool_use";
  } else if (choice.finish_reason === "length") {
    stopReason = "max_tokens";
  } else if (choice.finish_reason === "stop") {
    stopReason = "end_turn";
  } else if (choice.finish_reason === "content_filter") {
    stopReason = "end_turn";
  }

  return {
    id: `msg_${randomUUID().replace(/-/g, "")}`,
    type: "message",
    role: "assistant",
    content,
    model: requestModel || AZURE_MODEL,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: openaiResponse.usage?.prompt_tokens || 0,
      output_tokens: openaiResponse.usage?.completion_tokens || 0,
    },
  };
}

function sendSSE(res, event, data) {
  if (res.writableEnded) return;
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

async function handleStreaming(res, openaiBody, requestModel, apiKey) {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  const messageId = `msg_${randomUUID().replace(/-/g, "")}`;

  sendSSE(res, "message_start", {
    type: "message_start",
    message: {
      id: messageId,
      type: "message",
      role: "assistant",
      content: [],
      model: requestModel || AZURE_MODEL,
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    },
  });

  sendSSE(res, "ping", { type: "ping" });

  let currentContentIndex = -1;
  let hasTextBlock = false;
  let toolCallBlocks = {};
  let finishReason = null;
  let usageData = null;
  let fullText = "";
  let isDisconnected = false;

  res.on("close", () => { isDisconnected = true; });

  try {
    const fetchResponse = await fetch(AZURE_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(openaiBody),
    });

    if (!fetchResponse.ok) {
      const errText = await fetchResponse.text();
      addLog("error", `Streaming Error: ${fetchResponse.status}`, errText);
      console.error(`Azure API error: ${fetchResponse.status} ${errText}`);
      sendSSE(res, "content_block_start", {
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "" },
      });
      sendSSE(res, "content_block_delta", {
        type: "content_block_delta",
        index: 0,
        delta: {
          type: "text_delta",
          text: `Error from upstream: ${fetchResponse.status} - ${errText}`,
        },
      });
      sendSSE(res, "content_block_stop", {
        type: "content_block_stop",
        index: 0,
      });
      sendSSE(res, "message_delta", {
        type: "message_delta",
        delta: { stop_reason: "end_turn", stop_sequence: null },
        usage: { output_tokens: 0 },
      });
      sendSSE(res, "message_stop", { type: "message_stop" });
      res.end();
      return;
    }

    const reader = fetchResponse.body;
    let buffer = "";

    for await (const chunk of reader) {
      buffer += typeof chunk === "string" ? chunk : new TextDecoder().decode(chunk);
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed === "data: [DONE]") {
          if (trimmed === "data: [DONE]") {
            continue;
          }
          continue;
        }

        if (isDisconnected) break;
        if (!trimmed.startsWith("data: ")) continue;
        const jsonStr = trimmed.slice(6);

        let parsed;
        try {
          parsed = JSON.parse(jsonStr);
        } catch {
          continue;
        }

        if (parsed.usage) {
          usageData = parsed.usage;
        }

        const delta = parsed.choices?.[0]?.delta;
        const choiceFinish = parsed.choices?.[0]?.finish_reason;

        if (choiceFinish) {
          finishReason = choiceFinish;
        }

        if (!delta) continue;

        if (delta.content) {
          fullText += delta.content;
          if (!hasTextBlock) {
            currentContentIndex++;
            hasTextBlock = true;
            sendSSE(res, "content_block_start", {
              type: "content_block_start",
              index: currentContentIndex,
              content_block: { type: "text", text: "" },
            });
          }
          sendSSE(res, "content_block_delta", {
            type: "content_block_delta",
            index: currentContentIndex,
            delta: { type: "text_delta", text: delta.content },
          });
        }

        if (delta.tool_calls) {
          for (const tc of delta.tool_calls) {
            const tcIndex = tc.index ?? 0;

            if (!toolCallBlocks[tcIndex]) {
              if (hasTextBlock) {
                sendSSE(res, "content_block_stop", {
                  type: "content_block_stop",
                  index: currentContentIndex,
                });
                hasTextBlock = false;
              }
              currentContentIndex++;
              const toolId =
                tc.id || `toolu_${randomUUID().replace(/-/g, "")}`;
              toolCallBlocks[tcIndex] = {
                contentIndex: currentContentIndex,
                id: toolId,
                name: tc.function?.name || "",
                arguments: "",
              };
              sendSSE(res, "content_block_start", {
                type: "content_block_start",
                index: currentContentIndex,
                content_block: {
                  type: "tool_use",
                  id: toolId,
                  name: tc.function?.name || "",
                  input: {},
                },
              });
            }

            if (tc.function?.name && !toolCallBlocks[tcIndex].name) {
              toolCallBlocks[tcIndex].name = tc.function.name;
            }

            if (tc.function?.arguments) {
              toolCallBlocks[tcIndex].arguments += tc.function.arguments;
              sendSSE(res, "content_block_delta", {
                type: "content_block_delta",
                index: toolCallBlocks[tcIndex].contentIndex,
                delta: {
                  type: "input_json_delta",
                  partial_json: tc.function.arguments,
                },
              });
            }
          }
        }
      }
    }

    if (hasTextBlock) {
      sendSSE(res, "content_block_stop", {
        type: "content_block_stop",
        index: currentContentIndex,
      });
    }

    for (const tcIndex of Object.keys(toolCallBlocks)) {
      sendSSE(res, "content_block_stop", {
        type: "content_block_stop",
        index: toolCallBlocks[tcIndex].contentIndex,
      });
    }

    if (currentContentIndex === -1) {
      sendSSE(res, "content_block_start", {
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "" },
      });
      sendSSE(res, "content_block_stop", {
        type: "content_block_stop",
        index: 0,
      });
    }

    let stopReason = "end_turn";
    if (finishReason === "tool_calls") stopReason = "tool_use";
    else if (finishReason === "length") stopReason = "max_tokens";

    sendSSE(res, "message_delta", {
      type: "message_delta",
      delta: { stop_reason: stopReason, stop_sequence: null },
      usage: { output_tokens: usageData?.completion_tokens || 0 },
    });

    const finalResponse = {
      id: messageId,
      type: "message",
      role: "assistant",
      model: requestModel || AZURE_MODEL,
      content: [],
      stop_reason: stopReason,
      usage: {
        input_tokens: usageData?.prompt_tokens || 0,
        output_tokens: usageData?.completion_tokens || 0
      }
    };
    if (fullText) finalResponse.content.push({ type: "text", text: fullText });
    for (const tc of Object.values(toolCallBlocks)) {
      let input = {};
      try { input = JSON.parse(tc.arguments || "{}"); } catch(e) {}
      finalResponse.content.push({
        type: "tool_use",
        id: tc.id,
        name: tc.name,
        input
      });
    }

    addLog("response", "Full Streaming Response", finalResponse);

    sendSSE(res, "message_stop", { type: "message_stop" });
    res.end();
  } catch (err) {
    addLog("error", "Streaming Exception", err.message);
    console.error("Streaming error:", err);
    if (!res.writableEnded) {
      sendSSE(res, "error", {
        type: "error",
        error: { type: "api_error", message: err.message },
      });
      res.end();
    }
  }
}

app.post("/v1/messages", async (req, res) => {
  const anthropicBody = req.body;
  const apiKey = req.headers["x-api-key"] || AZURE_API_KEY;

  addLog("request", "Anthropic Request Received", anthropicBody);

  console.log(
    `[${new Date().toISOString()}] POST /v1/messages | model=${anthropicBody.model} stream=${!!anthropicBody.stream} tools=${anthropicBody.tools?.length || 0}`
  );

  if (!apiKey) {
    return res.status(401).json({
      type: "error",
      error: {
        type: "authentication_error",
        message: "No API key provided. Set AZURE_API_KEY env var or pass x-api-key header.",
      },
    });
  }

  const openaiBody = convertAnthropicToOpenAI(anthropicBody);
  addLog("info", "Converted OpenAI Body", openaiBody);

  if (anthropicBody.stream) {
    return handleStreaming(res, openaiBody, anthropicBody.model, apiKey);
  }

  try {
    const fetchResponse = await fetch(AZURE_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(openaiBody),
    });

    if (!fetchResponse.ok) {
      const errText = await fetchResponse.text();
      addLog("error", `Upstream Error: ${fetchResponse.status}`, errText);
      console.error(`Azure API error: ${fetchResponse.status} ${errText}`);
      return res.status(fetchResponse.status).json({
        type: "error",
        error: {
          type: "api_error",
          message: `Upstream error: ${errText}`,
        },
      });
    }

    const openaiResponse = await fetchResponse.json();
    addLog("response", "Upstream OpenAI Response", openaiResponse);
    const anthropicResponse = convertOpenAIToAnthropic(
      openaiResponse,
      anthropicBody.model
    );
    addLog("response", "Anthropic Response Sent", anthropicResponse);
    return res.json(anthropicResponse);
  } catch (err) {
    addLog("error", "Request Exception", err.message);
    console.error("Non-streaming error:", err);
    return res.status(500).json({
      type: "error",
      error: { type: "api_error", message: err.message },
    });
  }
});

app.get(["/models", "/v1/models"], (req, res) => {
  res.json({
    "data": [
      {
        "type": "model",
        "id": "claude-opus-4-7",
        "display_name": "Claude Opus 4.7",
        "created_at": "2026-04-14T00:00:00Z",
        "max_input_tokens": 1000000,
        "max_tokens": 128000,
        "capabilities": {
          "batch": {
            "supported": true
          },
          "citations": {
            "supported": true
          },
          "code_execution": {
            "supported": true
          },
          "context_management": {
            "supported": true,
            "clear_tool_uses_20250919": {
              "supported": true
            },
            "clear_thinking_20251015": {
              "supported": true
            },
            "compact_20260112": {
              "supported": true
            }
          },
          "effort": {
            "supported": true,
            "low": {
              "supported": true
            },
            "medium": {
              "supported": true
            },
            "high": {
              "supported": true
            },
            "max": {
              "supported": true
            }
          },
          "image_input": {
            "supported": true
          },
          "pdf_input": {
            "supported": true
          },
          "structured_outputs": {
            "supported": true
          },
          "thinking": {
            "supported": true,
            "types": {
              "enabled": {
                "supported": false
              },
              "adaptive": {
                "supported": true
              }
            }
          }
        }
      },
      {
        "type": "model",
        "id": "claude-sonnet-4-6",
        "display_name": "Claude Sonnet 4.6",
        "created_at": "2026-02-17T00:00:00Z",
        "max_input_tokens": 1000000,
        "max_tokens": 128000,
        "capabilities": {
          "batch": {
            "supported": true
          },
          "citations": {
            "supported": true
          },
          "code_execution": {
            "supported": true
          },
          "context_management": {
            "supported": true,
            "clear_tool_uses_20250919": {
              "supported": true
            },
            "clear_thinking_20251015": {
              "supported": true
            },
            "compact_20260112": {
              "supported": true
            }
          },
          "effort": {
            "supported": true,
            "low": {
              "supported": true
            },
            "medium": {
              "supported": true
            },
            "high": {
              "supported": true
            },
            "max": {
              "supported": true
            }
          },
          "image_input": {
            "supported": true
          },
          "pdf_input": {
            "supported": true
          },
          "structured_outputs": {
            "supported": true
          },
          "thinking": {
            "supported": true,
            "types": {
              "enabled": {
                "supported": true
              },
              "adaptive": {
                "supported": true
              }
            }
          }
        }
      },
      {
        "type": "model",
        "id": "claude-opus-4-6",
        "display_name": "Claude Opus 4.6",
        "created_at": "2026-02-04T00:00:00Z",
        "max_input_tokens": 1000000,
        "max_tokens": 128000,
        "capabilities": {
          "batch": {
            "supported": true
          },
          "citations": {
            "supported": true
          },
          "code_execution": {
            "supported": true
          },
          "context_management": {
            "supported": true,
            "clear_tool_uses_20250919": {
              "supported": true
            },
            "clear_thinking_20251015": {
              "supported": true
            },
            "compact_20260112": {
              "supported": true
            }
          },
          "effort": {
            "supported": true,
            "low": {
              "supported": true
            },
            "medium": {
              "supported": true
            },
            "high": {
              "supported": true
            },
            "max": {
              "supported": true
            }
          },
          "image_input": {
            "supported": true
          },
          "pdf_input": {
            "supported": true
          },
          "structured_outputs": {
            "supported": true
          },
          "thinking": {
            "supported": true,
            "types": {
              "enabled": {
                "supported": true
              },
              "adaptive": {
                "supported": true
              }
            }
          }
        }
      },
      {
        "type": "model",
        "id": "claude-opus-4-5-20251101",
        "display_name": "Claude Opus 4.5",
        "created_at": "2025-11-24T00:00:00Z",
        "max_input_tokens": 200000,
        "max_tokens": 64000,
        "capabilities": {
          "batch": {
            "supported": true
          },
          "citations": {
            "supported": true
          },
          "code_execution": {
            "supported": true
          },
          "context_management": {
            "supported": true,
            "clear_tool_uses_20250919": {
              "supported": true
            },
            "clear_thinking_20251015": {
              "supported": true
            },
            "compact_20260112": {
              "supported": false
            }
          },
          "effort": {
            "supported": true,
            "low": {
              "supported": true
            },
            "medium": {
              "supported": true
            },
            "high": {
              "supported": true
            },
            "max": {
              "supported": false
            }
          },
          "image_input": {
            "supported": true
          },
          "pdf_input": {
            "supported": true
          },
          "structured_outputs": {
            "supported": true
          },
          "thinking": {
            "supported": true,
            "types": {
              "enabled": {
                "supported": true
              },
              "adaptive": {
                "supported": false
              }
            }
          }
        }
      },
      {
        "type": "model",
        "id": "claude-haiku-4-5-20251001",
        "display_name": "Claude Haiku 4.5",
        "created_at": "2025-10-15T00:00:00Z",
        "max_input_tokens": 200000,
        "max_tokens": 64000,
        "capabilities": {
          "batch": {
            "supported": true
          },
          "citations": {
            "supported": true
          },
          "code_execution": {
            "supported": false
          },
          "context_management": {
            "supported": true,
            "clear_tool_uses_20250919": {
              "supported": true
            },
            "clear_thinking_20251015": {
              "supported": true
            },
            "compact_20260112": {
              "supported": false
            }
          },
          "effort": {
            "supported": false,
            "low": {
              "supported": false
            },
            "medium": {
              "supported": false
            },
            "high": {
              "supported": false
            },
            "max": {
              "supported": false
            }
          },
          "image_input": {
            "supported": true
          },
          "pdf_input": {
            "supported": true
          },
          "structured_outputs": {
            "supported": true
          },
          "thinking": {
            "supported": true,
            "types": {
              "enabled": {
                "supported": true
              },
              "adaptive": {
                "supported": false
              }
            }
          }
        }
      },
      {
        "type": "model",
        "id": "claude-sonnet-4-5-20250929",
        "display_name": "Claude Sonnet 4.5",
        "created_at": "2025-09-29T00:00:00Z",
        "max_input_tokens": 1000000,
        "max_tokens": 64000,
        "capabilities": {
          "batch": {
            "supported": true
          },
          "citations": {
            "supported": true
          },
          "code_execution": {
            "supported": true
          },
          "context_management": {
            "supported": true,
            "clear_tool_uses_20250919": {
              "supported": true
            },
            "clear_thinking_20251015": {
              "supported": true
            },
            "compact_20260112": {
              "supported": false
            }
          },
          "effort": {
            "supported": false,
            "low": {
              "supported": false
            },
            "medium": {
              "supported": false
            },
            "high": {
              "supported": false
            },
            "max": {
              "supported": false
            }
          },
          "image_input": {
            "supported": true
          },
          "pdf_input": {
            "supported": true
          },
          "structured_outputs": {
            "supported": true
          },
          "thinking": {
            "supported": true,
            "types": {
              "enabled": {
                "supported": true
              },
              "adaptive": {
                "supported": false
              }
            }
          }
        }
      },
      {
        "type": "model",
        "id": "claude-opus-4-1-20250805",
        "display_name": "Claude Opus 4.1",
        "created_at": "2025-08-05T00:00:00Z",
        "max_input_tokens": 200000,
        "max_tokens": 32000,
        "capabilities": {
          "batch": {
            "supported": true
          },
          "citations": {
            "supported": true
          },
          "code_execution": {
            "supported": false
          },
          "context_management": {
            "supported": true,
            "clear_tool_uses_20250919": {
              "supported": true
            },
            "clear_thinking_20251015": {
              "supported": true
            },
            "compact_20260112": {
              "supported": false
            }
          },
          "effort": {
            "supported": false,
            "low": {
              "supported": false
            },
            "medium": {
              "supported": false
            },
            "high": {
              "supported": false
            },
            "max": {
              "supported": false
            }
          },
          "image_input": {
            "supported": true
          },
          "pdf_input": {
            "supported": true
          },
          "structured_outputs": {
            "supported": true
          },
          "thinking": {
            "supported": true,
            "types": {
              "enabled": {
                "supported": true
              },
              "adaptive": {
                "supported": false
              }
            }
          }
        }
      },
      {
        "type": "model",
        "id": "claude-opus-4-20250514",
        "display_name": "Claude Opus 4",
        "created_at": "2025-05-22T00:00:00Z",
        "max_input_tokens": 200000,
        "max_tokens": 32000,
        "capabilities": {
          "batch": {
            "supported": true
          },
          "citations": {
            "supported": true
          },
          "code_execution": {
            "supported": false
          },
          "context_management": {
            "supported": true,
            "clear_tool_uses_20250919": {
              "supported": true
            },
            "clear_thinking_20251015": {
              "supported": true
            },
            "compact_20260112": {
              "supported": false
            }
          },
          "effort": {
            "supported": false,
            "low": {
              "supported": false
            },
            "medium": {
              "supported": false
            },
            "high": {
              "supported": false
            },
            "max": {
              "supported": false
            }
          },
          "image_input": {
            "supported": true
          },
          "pdf_input": {
            "supported": true
          },
          "structured_outputs": {
            "supported": false
          },
          "thinking": {
            "supported": true,
            "types": {
              "enabled": {
                "supported": true
              },
              "adaptive": {
                "supported": false
              }
            }
          }
        }
      },
      {
        "type": "model",
        "id": "claude-sonnet-4-20250514",
        "display_name": "Claude Sonnet 4",
        "created_at": "2025-05-22T00:00:00Z",
        "max_input_tokens": 1000000,
        "max_tokens": 64000,
        "capabilities": {
          "batch": {
            "supported": true
          },
          "citations": {
            "supported": true
          },
          "code_execution": {
            "supported": false
          },
          "context_management": {
            "supported": true,
            "clear_tool_uses_20250919": {
              "supported": true
            },
            "clear_thinking_20251015": {
              "supported": true
            },
            "compact_20260112": {
              "supported": false
            }
          },
          "effort": {
            "supported": false,
            "low": {
              "supported": false
            },
            "medium": {
              "supported": false
            },
            "high": {
              "supported": false
            },
            "max": {
              "supported": false
            }
          },
          "image_input": {
            "supported": true
          },
          "pdf_input": {
            "supported": true
          },
          "structured_outputs": {
            "supported": false
          },
          "thinking": {
            "supported": true,
            "types": {
              "enabled": {
                "supported": true
              },
              "adaptive": {
                "supported": false
              }
            }
          }
        }
      }
    ],
    "has_more": false,
    "first_id": "claude-opus-4-7",
    "last_id": "claude-sonnet-4-20250514"
  });
});

app.get("/logs", (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  const id = Date.now();
  clients.push({ id, res });
  logBuffer.forEach(l => res.write(`data: ${JSON.stringify(l)}\n\n`));
  req.on("close", () => { clients = clients.filter(c => c.id !== id); });
});

app.get("/", (req, res) => {
  const statusHtml = `
  <!DOCTYPE html>
  <html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Azure Foundry Proxy | Status</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
      :root {
        --bg: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.7);
        --accent: #38bdf8;
        --accent-glow: rgba(56, 189, 248, 0.3);
        --text: #f1f5f9;
        --text-muted: #94a3b8;
        --border: rgba(255, 255, 255, 0.1);
        --success: #10b981;
        --error: #ef4444;
        --request: #8b5cf6;
      }
      * { margin: 0; padding: 0; box-sizing: border-box; }
      body {
        font-family: 'Inter', sans-serif;
        background-color: var(--bg);
        background-image: 
          radial-gradient(circle at 0% 0%, rgba(56, 189, 248, 0.15) 0%, transparent 50%),
          radial-gradient(circle at 100% 100%, rgba(56, 189, 248, 0.1) 0%, transparent 50%);
        color: var(--text);
        min-height: 100vh;
        display: flex;
        justify-content: center;
        padding: 2rem 1rem;
      }
      .container {
        width: 100%;
        max-width: 900px;
        z-index: 1;
      }
      .card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        margin-bottom: 1.5rem;
      }
      .header { display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem; }
      .logo { width: 40px; height: 40px; background: var(--accent); border-radius: 10px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 15px var(--accent-glow); }
      .logo svg { width: 24px; height: 24px; fill: var(--bg); }
      h1 { font-size: 1.25rem; font-weight: 700; }
      .status-badge { display: inline-flex; align-items: center; gap: 0.5rem; background: rgba(16, 185, 129, 0.1); color: var(--success); padding: 0.4rem 0.8rem; border-radius: 99px; font-size: 0.75rem; font-weight: 600; margin-left: auto; }
      .status-dot { width: 6px; height: 6px; background: var(--success); border-radius: 50%; box-shadow: 0 0 8px var(--success); }
      
      .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
      .info-item { background: rgba(15, 23, 42, 0.4); padding: 1rem; border-radius: 12px; border: 1px solid var(--border); }
      .label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 0.25rem; }
      .value { font-family: 'Fira Code', monospace; font-size: 0.8rem; color: var(--accent); overflow: hidden; text-overflow: ellipsis; }

      .logs-container {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 20px;
        border: 1px solid var(--border);
        overflow: hidden;
        display: flex;
        flex-direction: column;
        height: 500px;
      }
      .logs-header { padding: 1rem 1.5rem; border-bottom: 1px solid var(--border); background: rgba(0,0,0,0.2); display: flex; justify-content: space-between; align-items: center; }
      .logs-title { font-size: 0.875rem; font-weight: 600; display: flex; align-items: center; gap: 0.5rem; }
      #logs-list { flex: 1; overflow-y: auto; padding: 1rem; font-family: 'Fira Code', monospace; font-size: 0.75rem; scroll-behavior: smooth; }
      .log-entry { margin-bottom: 0.75rem; padding-bottom: 0.75rem; border-bottom: 1px solid rgba(255,255,255,0.03); animation: fadeIn 0.3s ease; }
      @keyframes fadeIn { from { opacity: 0; transform: translateX(-5px); } to { opacity: 1; transform: translateX(0); } }
      .log-time { color: var(--text-muted); margin-right: 0.75rem; font-size: 0.7rem; }
      .log-type { padding: 2px 6px; border-radius: 4px; font-size: 0.65rem; font-weight: 700; margin-right: 0.75rem; text-transform: uppercase; }
      .type-request { background: rgba(139, 92, 246, 0.2); color: var(--request); }
      .type-response { background: rgba(16, 185, 129, 0.2); color: var(--success); }
      .type-error { background: rgba(239, 68, 68, 0.2); color: var(--error); }
      .log-msg { color: var(--text); }
      .log-data { display: block; background: rgba(0,0,0,0.3); padding: 0.5rem; border-radius: 6px; margin-top: 0.5rem; color: #cbd5e1; white-space: pre-wrap; font-size: 0.7rem; }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card">
        <div class="header">
          <div class="logo"><svg viewBox="0 0 24 24"><path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71L12 2z"/></svg></div>
          <div><h1>Azure Foundry Proxy</h1><p style="color: var(--text-muted); font-size: 0.75rem;">Claude Code Bridge</p></div>
          <div class="status-badge"><div class="status-dot"></div>Online</div>
        </div>
        <div class="grid">
          <div class="info-item"><div class="label">Target</div><div class="value">${AZURE_ENDPOINT}</div></div>
          <div class="info-item"><div class="label">Model</div><div class="value">${AZURE_MODEL}</div></div>
          <div class="info-item"><div class="label">Local</div><div class="value">http://localhost:${PORT}</div></div>
        </div>
      </div>

      <div class="logs-container">
        <div class="logs-header">
          <div class="logs-title">
            <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
            Live Activity Feed
          </div>
          <div style="font-size: 0.7rem; color: var(--text-muted);">Real-time SSE Stream</div>
        </div>
        <div id="logs-list"></div>
      </div>
    </div>

    <script>
      const logsList = document.getElementById('logs-list');
      const eventSource = new EventSource('/logs');

      eventSource.onmessage = (event) => {
        const log = JSON.parse(event.data);
        const div = document.createElement('div');
        div.className = 'log-entry';
        
        let dataHtml = '';
        if (log.data) {
          dataHtml = \`<pre class="log-data">\${log.data}</pre>\`;
        }

        div.innerHTML = \`
          <span class="log-time">\${log.timestamp}</span>
          <span class="log-type type-\${log.type}">\${log.type}</span>
          <span class="log-msg">\${log.message}</span>
          \${dataHtml}
        \`;
        
        logsList.appendChild(div);
        logsList.scrollTop = logsList.scrollHeight;
        
        if (logsList.children.length > 100) {
          logsList.removeChild(logsList.firstChild);
        }
      };

      eventSource.onerror = () => {
        console.error("SSE Connection lost. Reconnecting...");
      };
    </script>
  </body>
  </html>
  `;
  res.send(statusHtml);
});

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

app.listen(PORT, () => {
  console.log(`\n╔══════════════════════════════════════════════════╗`);
  console.log(`║    Anthropic → Azure OpenAI Proxy                ║`);
  console.log(`╠══════════════════════════════════════════════════╣`);
  console.log(`║  Listening:  http://localhost:${PORT}              ║`);
  console.log(`║  Endpoint:   ${AZURE_ENDPOINT.substring(0, 38)}...║`);
  console.log(`║  Model:      ${AZURE_MODEL.padEnd(36)}║`);
  console.log(`║  API Key:    ${AZURE_API_KEY ? "✓ configured" : "✗ missing (set AZURE_API_KEY)".padEnd(36)}║`);
  console.log(`╚══════════════════════════════════════════════════╝`);
  console.log(`\nSet in Claude Code:`);
  console.log(`  ANTHROPIC_BASE_URL=http://localhost:${PORT}`);
  console.log(``);
});
