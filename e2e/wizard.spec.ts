import { expect, test, type Page } from "@playwright/test";
import type { LibraryData, WizardReply } from "../src/lib/contracts";
const library: LibraryData = {
  collector: {
    id: "wizard-fixture",
    name: "Wizard listener",
    email: "wizard@example.test",
    isOwner: false,
  },
  records: [],
  crates: [],
  sharedCrates: [],
  listens: [],
};
const album = {
  masterId: 8883,
  artist: "Michael Jackson",
  title: "Thriller",
  originalYear: 1982,
  coverUrl: null,
  sourceUrl: "https://www.discogs.com/master/8883",
};
const reply: WizardReply = {
  reply:
    "Start with Thriller for dance-floor grooves and pop hooks. Off the Wall is a great next stop for disco and soul.",
  recordIds: [],
  recommendations: [
    {
      ...album,
      reason: "A strong first buy for dance-floor grooves and pop hooks.",
    },
  ],
  sources: [
    {
      id: "master:8883",
      kind: "master",
      title: "Michael Jackson — Thriller",
      url: album.sourceUrl,
    },
  ],
  proposal: null,
};
async function setup(page: Page) {
  await page.route("**/api/library", (route) =>
    route.fulfill({ json: library }),
  );
}
test("buying advice works with an empty library and leads into real pressing exploration", async ({
  page,
}) => {
  await setup(page);
  const questions: {
    message: string;
    history: { role: string; content: string }[];
  }[] = [];
  await page.route("**/api/wizard", (route) => {
    questions.push(route.request().postDataJSON());
    return route.fulfill({ json: reply });
  });
  await page.route("**/api/catalog/albums**", (route) => {
    const path = new URL(route.request().url()).pathname;
    return route.fulfill({
      json: path.endsWith("/versions")
        ? {
            candidates: [],
            pagination: { page: 1, pages: 0, total: 0, hasMore: false },
          }
        : path.endsWith("/8883")
          ? album
          : {
              albums: [album],
              pagination: { page: 1, pages: 1, total: 1, hasMore: false },
            },
    });
  });
  await page.goto("/wizard");
  await page
    .getByRole("button", {
      name: "Which Michael Jackson album should I buy first, and why?",
      exact: true,
    })
    .click();
  await page.getByRole("button", { name: "Ask Wizard", exact: true }).click();
  await expect(
    page.getByRole("region", { name: "Albums to explore" }),
  ).toContainText("Thriller");
  await expect(
    page.getByRole("region", { name: "From your collection" }),
  ).toHaveCount(0);
  await expect(
    page
      .getByRole("navigation", { name: "Sources checked for this answer" })
      .getByRole("link"),
  ).toHaveAttribute("href", album.sourceUrl);
  await expect(
    page.getByRole("link", { name: "Discogs album details" }),
  ).toHaveAttribute("href", "https://www.discogs.com/master/8883");
  await page
    .getByLabel("Ask Wax Wizard")
    .fill("Which pressing should I look for?");
  await page.getByRole("button", { name: "Ask Wizard", exact: true }).click();
  await expect.poll(() => questions.length).toBe(2);
  expect(questions[1].history).toEqual([
    {
      role: "user",
      content: "Which Michael Jackson album should I buy first, and why?",
    },
    { role: "assistant", content: reply.reply },
  ]);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth + 1,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: test.info().outputPath("wizard-discovery.png"),
    fullPage: true,
  });
  await page
    .getByRole("link", { name: "Explore pressings", exact: true })
    .last()
    .click();
  await expect(page).toHaveURL(/\/add\?.*artist=Michael/);
  await expect(
    page.getByText("Thriller", { exact: true }).first(),
  ).toBeVisible();
});

test("a failed Wizard request preserves the question for retry and sends no collection writes", async ({
  page,
}) => {
  await setup(page);
  let attempts = 0;
  const writes: string[] = [];
  await page.route("**/api/records**", (route) => {
    writes.push(route.request().method());
    return route.abort();
  });
  await page.route("**/api/wizard", (route) => {
    attempts++;
    return route.fulfill(
      attempts === 1
        ? {
            status: 503,
            json: { error: "The Wizard is taking a break. Try again shortly." },
          }
        : { json: reply },
    );
  });
  await page.goto("/wizard");
  await page
    .getByLabel("Ask Wax Wizard")
    .fill("I want to buy a Michael Jackson record.");
  await page.getByRole("button", { name: "Ask Wizard", exact: true }).click();
  await expect(
    page.getByRole("alert").filter({ hasText: "taking a break" }),
  ).toBeVisible();
  await expect(page.getByLabel("Ask Wax Wizard")).toHaveValue(
    "I want to buy a Michael Jackson record.",
  );
  await page.getByRole("button", { name: "Ask Wizard", exact: true }).click();
  await expect(
    page.getByRole("region", { name: "Albums to explore" }),
  ).toContainText("Thriller");
  expect(writes).toEqual([]);
});

test("an answer does not erase a new question typed while waiting", async ({
  page,
}) => {
  await setup(page);
  let release: (() => void) | undefined;
  await page.route("**/api/wizard", async (route) => {
    await new Promise<void>((resolve) => {
      release = resolve;
    });
    await route.fulfill({ json: reply });
  });
  await page.goto("/wizard");
  await page
    .getByLabel("Ask Wax Wizard")
    .fill("Recommend a Michael Jackson record.");
  await page.getByRole("button", { name: "Ask Wizard", exact: true }).click();
  await expect(page.getByRole("status")).toContainText(
    "Digging into your question",
  );
  await page
    .getByLabel("Ask Wax Wizard")
    .fill("Now tell me about Curtis Mayfield.");
  await expect.poll(() => Boolean(release)).toBeTruthy();
  release!();
  await expect(
    page.getByRole("region", { name: "Albums to explore" }),
  ).toContainText("Thriller");
  await expect(page.getByLabel("Ask Wax Wizard")).toHaveValue(
    "Now tell me about Curtis Mayfield.",
  );
});
