import { serve } from "@hono/node-server";
import { createApp } from "./app.js";
import { config } from "./config.js";
import { Store } from "./store.js";

const store = new Store(config.dataFile);
const app = createApp(store);

serve({ fetch: app.fetch, port: config.port }, (info) => {
  console.log(`qav-api listening on http://localhost:${info.port}`);
  console.log(`  LiveKit: ${config.livekit.url}  agent: ${config.livekit.engineAgentName}`);
  if (config.apiKey === "qav_dev_key_change_me") {
    console.warn("  QAV_API_KEY is the dev default — change it before exposing this API.");
  }
});
