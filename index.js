require("dotenv").config();
const express = require("express");
const { randomUUID } = require("crypto");

const app = express();
app.use(express.json({ limit: "50mb" }));


const AZURE_ENDPOINT = process.env.AZURE_ENDPOINT;
const AZURE_API_KEY = process.env.AZURE_API_KEY;
let AZURE_MODEL = process.env.AZURE_MODEL || "DeepSeek-V4-Flash";
const PORT = process.env.PORT || 8082;

let clients = [];
let logBuffer = [];
const MAX_LOGS = 100;
let totalInputTokensPro = 0;
let totalOutputTokensPro = 0;
let totalInputTokensFlash = 0;
let totalOutputTokensFlash = 0;

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

    if (usageData) {
      if (AZURE_MODEL.toLowerCase().includes("pro")) {
        totalInputTokensPro += usageData.prompt_tokens || 0;
        totalOutputTokensPro += usageData.completion_tokens || 0;
      } else {
        totalInputTokensFlash += usageData.prompt_tokens || 0;
        totalOutputTokensFlash += usageData.completion_tokens || 0;
      }
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
    if (anthropicResponse.usage) {
      if (AZURE_MODEL.toLowerCase().includes("pro")) {
        totalInputTokensPro += anthropicResponse.usage.input_tokens || 0;
        totalOutputTokensPro += anthropicResponse.usage.output_tokens || 0;
      } else {
        totalInputTokensFlash += anthropicResponse.usage.input_tokens || 0;
        totalOutputTokensFlash += anthropicResponse.usage.output_tokens || 0;
      }
    }
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

app.get("/stats", (req, res) => {
  const inputCostPro = (totalInputTokensPro / 1000000) * 1.74;
  const outputCostPro = (totalOutputTokensPro / 1000000) * 3.48;
  const inputCostFlash = (totalInputTokensFlash / 1000000) * 0.19;
  const outputCostFlash = (totalOutputTokensFlash / 1000000) * 0.51;
  const totalCost = inputCostPro + outputCostPro + inputCostFlash + outputCostFlash;
  res.json({
    activeModel: AZURE_MODEL,
    inputTokensPro: totalInputTokensPro,
    outputTokensPro: totalOutputTokensPro,
    inputTokensFlash: totalInputTokensFlash,
    outputTokensFlash: totalOutputTokensFlash,
    totalCost: totalCost.toFixed(6)
  });
});

app.post("/reset-stats", (req, res) => {
  totalInputTokensPro = 0;
  totalOutputTokensPro = 0;
  totalInputTokensFlash = 0;
  totalOutputTokensFlash = 0;
  res.json({ success: true });
});

app.post("/change-model", (req, res) => {
  const { model } = req.body;
  if (model === "DeepSeek-V4-Pro" || model === "DeepSeek-V4-Flash") {
    AZURE_MODEL = model;
    res.json({ success: true, model: AZURE_MODEL });
  } else {
    res.status(400).json({ error: "Invalid model selection" });
  }
});

app.get("/", (req, res) => {
  const statusHtml = `
  <!DOCTYPE html>
  <html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Azure Foundry Proxy | Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
      :root {
        --bg: #0b0f19;
        --card-bg: rgba(17, 24, 39, 0.7);
        --accent-cyan: #06b6d4;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --text: #f3f4f6;
        --text-muted: #9ca3af;
        --border: rgba(255, 255, 255, 0.08);
        --glow-cyan: rgba(6, 182, 212, 0.15);
        --glow-purple: rgba(139, 92, 246, 0.15);
        --success: #10b981;
      }
      * { margin: 0; padding: 0; box-sizing: border-box; }
      body {
        font-family: 'Inter', sans-serif;
        background-color: var(--bg);
        background-image: 
          radial-gradient(circle at 10% 20%, rgba(6, 182, 212, 0.12) 0%, transparent 40%),
          radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.15) 0%, transparent 40%);
        color: var(--text);
        min-height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem 1rem;
      }
      .container {
        width: 100%;
        max-width: 800px;
        z-index: 1;
      }
      .card {
        background: var(--card-bg);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
      }
      .header { 
        display: flex; 
        align-items: center; 
        gap: 1.25rem; 
        margin-bottom: 2.5rem; 
        border-bottom: 1px solid var(--border);
        padding-bottom: 1.5rem;
      }
      .logo { 
        width: 48px; 
        height: 48px; 
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue)); 
        border-radius: 12px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        box-shadow: 0 0 20px var(--glow-cyan); 
      }
      .logo svg { width: 28px; height: 28px; fill: var(--bg); }
      h1 { font-size: 1.5rem; font-weight: 700; background: linear-gradient(to right, #ffffff, #e2e8f0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
      .status-badge { 
        display: inline-flex; 
        align-items: center; 
        gap: 0.5rem; 
        background: rgba(16, 185, 129, 0.1); 
        color: var(--success); 
        padding: 0.5rem 1rem; 
        border-radius: 99px; 
        font-size: 0.8rem; 
        font-weight: 600; 
        margin-left: auto;
        border: 1px solid rgba(16, 185, 129, 0.2);
      }
      .status-dot { width: 8px; height: 8px; background: var(--success); border-radius: 50%; box-shadow: 0 0 10px var(--success); }
      
      .config-grid { 
        display: grid; 
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); 
        gap: 1.25rem; 
        margin-bottom: 2.5rem; 
      }
      .config-item { 
        background: rgba(255, 255, 255, 0.02); 
        padding: 1.25rem; 
        border-radius: 16px; 
        border: 1px solid var(--border); 
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }
      .config-item:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(6, 182, 212, 0.3);
        transform: translateY(-2px);
      }
      .label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 0.5rem; }
      .value { font-family: 'Fira Code', monospace; font-size: 0.85rem; color: var(--accent-cyan); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

      .model-switch-group {
        display: flex;
        background: rgba(0, 0, 0, 0.25);
        padding: 0.25rem;
        border-radius: 12px;
        border: 1px solid var(--border);
        margin-top: 0.25rem;
        width: 100%;
      }
      .model-btn {
        flex: 1;
        background: transparent;
        border: none;
        color: var(--text-muted);
        padding: 0.5rem;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        font-family: 'Inter', sans-serif;
      }
      .model-btn.active {
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
        color: var(--bg);
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.35);
      }
      .model-btn:hover:not(.active) {
        color: #fff;
        background: rgba(255, 255, 255, 0.03);
      }

      .stats-section {
        background: rgba(255, 255, 255, 0.01);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
      }
      .stats-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        color: #fff;
      }
      .stats-title svg {
        color: var(--accent-purple);
      }
      .cost-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(6, 182, 212, 0.1));
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
      }
      .cost-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.08) 0%, transparent 60%);
        pointer-events: none;
      }
      .cost-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
      }
      .cost-value {
        font-family: 'Fira Code', monospace;
        font-size: 3.5rem;
        font-weight: 700;
        color: #fff;
        text-shadow: 0 0 30px rgba(139, 92, 246, 0.4);
        line-height: 1;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
      }
      .cost-pricing {
        font-size: 0.75rem;
        color: var(--text-muted);
      }

      .tokens-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.25rem;
        margin-bottom: 1.5rem;
      }
      .token-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
      }
      .token-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
      }
      
      .token-details {
        margin-top: 0.75rem;
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
        font-family: 'Fira Code', monospace;
        font-size: 0.8rem;
      }
      .detail-label {
        color: var(--text-muted);
        font-size: 0.75rem;
      }
      .detail-val {
        color: #fff;
        font-weight: 500;
      }
      .token-cost-share {
        font-size: 0.75rem;
        color: var(--accent-cyan);
        font-family: 'Fira Code', monospace;
        margin-top: 0.25rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        padding-top: 0.4rem;
      }

      .bar-container {
        background: rgba(255, 255, 255, 0.05);
        height: 8px;
        border-radius: 99px;
        overflow: hidden;
        display: flex;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
      }
      .bar-input {
        background: var(--accent-purple);
        height: 100%;
        transition: width 0.5s ease;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
      }
      .bar-output {
        background: var(--accent-cyan);
        height: 100%;
        transition: width 0.5s ease;
        box-shadow: 0 0 10px rgba(6, 182, 212, 0.5);
      }

      .footer-actions {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 1rem;
      }
      .btn {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        cursor: pointer;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
      }
      .btn:hover {
        background: rgba(239, 68, 68, 0.2);
        border-color: rgba(239, 68, 68, 0.4);
        color: #fee2e2;
        transform: translateY(-1px);
      }
      .btn:active {
        transform: translateY(0);
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card">
        <div class="header">
          <div class="logo">
            <svg viewBox="0 0 24 24"><path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71L12 2z"/></svg>
          </div>
          <div>
            <h1>Azure Foundry Proxy</h1>
            <p style="color: var(--text-muted); font-size: 0.8rem;">Claude Code Bridge</p>
          </div>
          <div class="status-badge">
            <div class="status-dot"></div>Online
          </div>
        </div>

        <div class="config-grid">
          <div class="config-item">
            <div class="label">Target Endpoint</div>
            <div class="value" title="${AZURE_ENDPOINT}">${AZURE_ENDPOINT}</div>
          </div>
          <div class="config-item">
            <div class="label">Active Model</div>
            <div class="model-switch-group">
              <button class="model-btn" id="pro-btn" data-model="DeepSeek-V4-Pro">Pro</button>
              <button class="model-btn" id="flash-btn" data-model="DeepSeek-V4-Flash">Flash</button>
            </div>
          </div>
          <div class="config-item">
            <div class="label">Local API URL</div>
            <div class="value">http://localhost:${PORT}</div>
          </div>
        </div>

        <div class="stats-section">
          <div class="stats-title">
            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 002 2h2a2 2 0 002-2z" />
            </svg>
            Token Usage & Costs
          </div>

          <div class="cost-card">
            <div class="cost-label">Total Cost</div>
            <div class="cost-value" id="cost-display">$0.000000</div>
            <div class="cost-pricing" style="display: flex; justify-content: center; gap: 1.5rem; margin-top: 0.5rem; font-size: 0.75rem;">
              <span><strong>Pro:</strong> $1.74 / $3.48 (1M)</span>
              <span><strong>Flash:</strong> $0.19 / $0.51 (1M)</span>
            </div>
          </div>

          <div class="tokens-grid">
            <div class="token-card">
              <div class="token-title" style="color: var(--accent-purple); font-weight: 700;">DeepSeek V4 Pro</div>
              <div class="token-details">
                <div><span class="detail-label">Input:</span> <span class="detail-val" id="pro-input-display">0</span></div>
                <div><span class="detail-label">Output:</span> <span class="detail-val" id="pro-output-display">0</span></div>
                <div class="token-cost-share" id="pro-cost-display">$0.000000</div>
              </div>
            </div>
            <div class="token-card">
              <div class="token-title" style="color: var(--accent-cyan); font-weight: 700;">DeepSeek V4 Flash</div>
              <div class="token-details">
                <div><span class="detail-label">Input:</span> <span class="detail-val" id="flash-input-display">0</span></div>
                <div><span class="detail-label">Output:</span> <span class="detail-val" id="flash-output-display">0</span></div>
                <div class="token-cost-share" id="flash-cost-display">$0.000000</div>
              </div>
            </div>
          </div>

          <div class="bar-container">
            <div class="bar-input" id="pro-bar" style="width: 50%"></div>
            <div class="bar-output" id="flash-bar" style="width: 50%"></div>
          </div>

          <div class="footer-actions">
            <button class="btn" id="btn-reset">
              <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
              Reset Metrics
            </button>
          </div>
        </div>
      </div>
    </div>

    <script>
      const costDisplay = document.getElementById('cost-display');
      const proInputDisplay = document.getElementById('pro-input-display');
      const proOutputDisplay = document.getElementById('pro-output-display');
      const flashInputDisplay = document.getElementById('flash-input-display');
      const flashOutputDisplay = document.getElementById('flash-output-display');
      const proCostDisplay = document.getElementById('pro-cost-display');
      const flashCostDisplay = document.getElementById('flash-cost-display');
      const proBar = document.getElementById('pro-bar');
      const flashBar = document.getElementById('flash-bar');
      const btnReset = document.getElementById('btn-reset');
      const proBtn = document.getElementById('pro-btn');
      const flashBtn = document.getElementById('flash-btn');

      function formatNumber(num) {
        return num.toLocaleString();
      }

      async function fetchStats() {
        try {
          const res = await fetch('/stats');
          const data = await res.json();
          
          costDisplay.textContent = '$' + parseFloat(data.totalCost).toFixed(6);
          
          proInputDisplay.textContent = formatNumber(data.inputTokensPro);
          proOutputDisplay.textContent = formatNumber(data.outputTokensPro);
          flashInputDisplay.textContent = formatNumber(data.inputTokensFlash);
          flashOutputDisplay.textContent = formatNumber(data.outputTokensFlash);
          
          const proCost = (data.inputTokensPro / 1000000) * 1.74 + (data.outputTokensPro / 1000000) * 3.48;
          const flashCost = (data.inputTokensFlash / 1000000) * 0.19 + (data.outputTokensFlash / 1000000) * 0.51;
          
          proCostDisplay.textContent = '$' + proCost.toFixed(6);
          flashCostDisplay.textContent = '$' + flashCost.toFixed(6);
          
          const totalPro = data.inputTokensPro + data.outputTokensPro;
          const totalFlash = data.inputTokensFlash + data.outputTokensFlash;
          const total = totalPro + totalFlash;
          
          if (total > 0) {
            const proPercent = (totalPro / total) * 100;
            const flashPercent = (totalFlash / total) * 100;
            proBar.style.width = proPercent + '%';
            flashBar.style.width = flashPercent + '%';
          } else {
            proBar.style.width = '50%';
            proBar.style.backgroundColor = 'var(--accent-purple)';
            flashBar.style.width = '50%';
            flashBar.style.backgroundColor = 'var(--accent-cyan)';
          }
          
          if (data.activeModel === 'DeepSeek-V4-Pro') {
            proBtn.classList.add('active');
            flashBtn.classList.remove('active');
          } else {
            flashBtn.classList.add('active');
            proBtn.classList.remove('active');
          }
        } catch (err) {
          console.error('Error fetching stats:', err);
        }
      }

      async function setModel(modelName) {
        try {
          await fetch('/change-model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: modelName })
          });
          fetchStats();
        } catch (err) {
          console.error('Error setting model:', err);
        }
      }

      proBtn.addEventListener('click', () => setModel('DeepSeek-V4-Pro'));
      flashBtn.addEventListener('click', () => setModel('DeepSeek-V4-Flash'));

      btnReset.addEventListener('click', async () => {
        if (confirm('Are you sure you want to reset all token and cost statistics?')) {
          try {
            await fetch('/reset-stats', { method: 'POST' });
            fetchStats();
          } catch (err) {
            console.error('Error resetting stats:', err);
          }
        }
      });

      setInterval(fetchStats, 1000);
      fetchStats();
    </script>
  </body>
  </html>
  `;
  res.send(statusHtml);
});

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

if (!process.env.VERCEL) {
  app.listen(PORT, () => {
    console.log(`\n╔══════════════════════════════════════════════════╗`);
    console.log(`║    Anthropic → Azure OpenAI Proxy                ║`);
    console.log(`╠══════════════════════════════════════════════════╣`);
    console.log(`║  Listening:  http://localhost:${PORT}              ║`);
    console.log(`║  Endpoint:   ${(AZURE_ENDPOINT || "").substring(0, 38)}...║`);
    console.log(`║  Model:      ${AZURE_MODEL.padEnd(36)}║`);
    console.log(`║  API Key:    ${AZURE_API_KEY ? "✓ configured" : "✗ missing (set AZURE_API_KEY)".padEnd(36)}║`);
    console.log(`╚══════════════════════════════════════════════════╝`);
    console.log(`\nSet in Claude Code:`);
    console.log(`  ANTHROPIC_BASE_URL=http://localhost:${PORT}`);
    console.log(``);
  });
}

module.exports = app;