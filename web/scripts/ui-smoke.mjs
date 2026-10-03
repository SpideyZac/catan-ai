// End-to-end UI smoke test against a running server (default http://127.0.0.1:8000).
//
//   uv run catan-server --port 8000 &
//   CHROME_PATH="/path/to/chrome" npm run ui-smoke [-- --out shots/]
//
// Walks: home -> Play vs AI setup (3 players) -> setup placements -> roll -> compose a
// trade offer, then Pass & Play lobby -> curtain, and checks that background music plays. Fails on any page error or missing step and
// optionally saves screenshots.
import { chromium } from "playwright-core";
import { mkdirSync } from "node:fs";

const BASE = process.env.BASE_URL ?? "http://127.0.0.1:8000";
const outIdx = process.argv.indexOf("--out");
const OUT = outIdx > 0 ? process.argv[outIdx + 1] : null;
if (OUT) mkdirSync(OUT, { recursive: true });
const executablePath =
  process.env.CHROME_PATH ??
  (process.platform === "win32" ? "C:/Program Files/Google/Chrome/Application/chrome.exe" : undefined);

const browser = await chromium.launch({ executablePath });
const errors = [];
const shot = async (page, name) => OUT && page.screenshot({ path: `${OUT}/${name}.png` });

async function playSetup(page) {
  for (let i = 0; i < 300; i++) {
    if (await page.locator("button.roll").count()) return;
    if (await page.locator(".curtain button").count()) await page.click(".curtain button");
    const v = page.locator(".target-vertex");
    const e = page.locator(".target-edge");
    if (await v.count()) await v.nth(Math.floor(Math.random() * (await v.count()))).click({ force: true });
    else if (await e.count()) await e.first().click({ force: true });
    await page.waitForTimeout(250);
  }
  throw new Error("setup did not reach the roll phase");
}

try {
  // 1. Solo game vs AI, through to a trade offer.
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(BASE);
  await shot(page, "home");
  await page.fill("input", "Smoke");
  await page.click("text=Play vs AI");
  await page.waitForSelector(".quick-setup");
  const rules = page.locator(".quick-setup select");
  await rules.nth(0).selectOption("3"); // players
  await rules.nth(1).selectOption("8"); // points to win
  await shot(page, "quick-setup");
  await page.click(".quick-setup >> text=Start game");
  await page.waitForSelector(".board", { timeout: 15000 });
  const panels = await page.locator(".players .ppanel").count();
  if (panels !== 3) throw new Error(`expected 3 players from the setup dialog, got ${panels}`);
  await playSetup(page);
  await shot(page, "setup");
  await page.click("button.roll", { force: true });
  await page.waitForTimeout(1500);
  for (let i = 0; i < 20 && (await page.locator(".target-hex").count()); i++) {
    await page.locator(".target-hex").first().click({ force: true });
    await page.waitForTimeout(400);
    if (await page.locator(".victim").count()) await page.locator(".victim").first().click();
  }
  await page.click(".tab >> text=Trade");
  const have = (await page.locator(".hand .rcard .rcard-count").allInnerTexts()).map(Number);
  const give = have.findIndex((n) => n > 0);
  if (give >= 0) {
    const steppers = page.locator(".stepper");
    await steppers.nth(0).locator(".step").nth(give).locator(".step-btn").first().click();
    await steppers.nth(1).locator(".step").nth((give + 1) % 5).locator(".step-btn").first().click();
    await page.click("text=Offer to table");
    await page.waitForTimeout(800);
  }
  await shot(page, "trade");
  await page.close();

  // 2. Pass & play lobby and curtain.
  const p2 = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  p2.on("pageerror", (e) => errors.push(String(e)));
  await p2.goto(BASE);
  await p2.fill("input", "Parent");
  await p2.click("text=Pass & Play");
  await p2.waitForSelector(".lobby");
  await p2.waitForTimeout(500);
  await p2.locator(".seat").nth(1).locator("input").fill("Kid");
  await p2.locator(".seat").nth(1).locator("text=Add").click();
  for (const i of [2, 3]) await p2.locator(".seat").nth(i).locator("select").selectOption("heuristic");
  await p2.waitForTimeout(400);
  await shot(p2, "lobby");
  await p2.click("text=Start game");
  await p2.waitForSelector(".curtain", { timeout: 15000 });
  await shot(p2, "curtain");
  await p2.close();

  // 3. Background music: with effects muted, any oscillator started comes from the music.
  const p3 = await browser.newPage();
  p3.on("pageerror", (e) => errors.push(String(e)));
  await p3.addInitScript(() => {
    localStorage.setItem("catan.muted", "1");
    window.__osc = 0;
    const start = OscillatorNode.prototype.start;
    OscillatorNode.prototype.start = function (...a) {
      window.__osc++;
      return start.apply(this, a);
    };
  });
  await p3.goto(BASE);
  await p3.click("h1"); // user gesture unlocks the AudioContext
  await p3.waitForTimeout(2500);
  const playing = await p3.evaluate(() => window.__osc);
  await p3.click(".music-toggle");
  await p3.waitForTimeout(500);
  const before = await p3.evaluate(() => window.__osc);
  await p3.waitForTimeout(1500);
  const after = await p3.evaluate(() => window.__osc);
  await p3.click(".music-toggle"); // restore the default for later runs in this profile
  if (!playing) throw new Error("lobby music did not start");
  if (after !== before) throw new Error("music kept playing after being turned off");
  await p3.close();
} finally {
  await browser.close();
}

if (errors.length) {
  console.error("page errors:", errors);
  process.exit(1);
}
console.log("ui smoke: ok");
