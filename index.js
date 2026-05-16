require("dotenv").config();
const express = require("express");
const { randomUUID } = require("crypto");

const app = express();
app.use(express.json({ limit: "50mb" }));


const AZURE_ENDPOINT = process.env.AZURE_ENDPOINT;
const AZURE_API_KEY = process.env.AZURE_API_KEY;
const AZURE_MODEL = process.env.AZURE_MODEL || "DeepSeek-V4-Pro";
const PORT = process.env.PORT || 8082;

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

        for (const block of msg.content) {
          if (block.type === "text") {
            textParts.push(block.text);
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
            if (typeof block.content === "string") {
              resultContent = block.content;
            } else if (Array.isArray(block.content)) {
              resultContent = block.content
                .map((c) => (c.type === "text" ? c.text : JSON.stringify(c)))
                .join("\n");
            }
            toolResults.push({
              role: "tool",
              tool_call_id: block.tool_use_id,
              content: resultContent,
            });
          } else if (block.type === "image") {
            textParts.push("[image]");
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
            content: textParts.join("\n") || "",
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

    sendSSE(res, "message_stop", { type: "message_stop" });
    res.end();
  } catch (err) {
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
    const anthropicResponse = convertOpenAIToAnthropic(
      openaiResponse,
      anthropicBody.model
    );
    return res.json(anthropicResponse);
  } catch (err) {
    console.error("Non-streaming error:", err);
    return res.status(500).json({
      type: "error",
      error: { type: "api_error", message: err.message },
    });
  }
});

app.get("/v1/models", (req, res) => {
  res.json({
    data: [
      {
        id: AZURE_MODEL,
        object: "model",
        created: Date.now(),
        owned_by: "azure",
      },
    ],
  });
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
