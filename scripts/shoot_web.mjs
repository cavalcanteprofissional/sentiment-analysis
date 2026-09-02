// Ferramenta de screenshot para paginas web (dev apenas, fora do runtime do projeto).
// Requer: npm i playwright && npx playwright install chromium
// Uso:
//   node scripts/shoot_web.mjs <config.json>
// Config:
// {
//   "url": "http://localhost:5000",
//   "out": "docs/prints/00_mlflow_ui.png",
//   "waitMs": 3000,
//   "fullPage": false,
//   "viewport": { "width": 1600, "height": 1000 },
//   "actions": [
//     { "type": "fill", "selector": "textarea", "value": "texto a digitar" },
//     { "type": "click", "selector": "button" },
//     { "type": "wait", "ms": 4000 }
//   ]
// }
import { chromium } from "playwright";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const configPath = process.argv[2];
if (!configPath) {
  console.error("Uso: node scripts/shoot_web.mjs <config.json>");
  process.exit(1);
}

const config = JSON.parse(await readFile(configPath, "utf8"));
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: config.viewport ?? { width: 1600, height: 1000 },
});

try {
  await page.goto(config.url, { waitUntil: "domcontentloaded", timeout: 60000 });
  for (const action of config.actions ?? []) {
    if (action.type === "fill") {
      await page.locator(action.selector).first().fill(action.value ?? "");
    } else if (action.type === "click") {
      await page.locator(action.selector).first().click();
    } else if (action.type === "wait") {
      await page.waitForTimeout(action.ms ?? 2000);
    }
  }
  await page.waitForTimeout(config.waitMs ?? 2000);
  const out = path.resolve(config.out);
  await page.screenshot({ path: out, fullPage: config.fullPage ?? false });
  console.log(`Web-shot salvo: ${out}`);
} finally {
  await browser.close();
}