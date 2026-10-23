import { expect, test, type Page } from "@playwright/test";
import {
  DEFAULT_COPY,
  type CatalogCandidate,
  type LibraryData,
} from "../src/lib/contracts";
import type {
  ExplorerArtist,
  ExplorerCredit,
  ExplorerDiscography,
  ExplorerRelease,
  ExplorerStart,
} from "../src/lib/explorer";

const copyId = "00000000-0000-4000-8000-000000000101";
const kindOfBlueId = 1000;
const wyntonId = 101;
const sextetId = 202;
const kellyBlueId = 2000;
const undilutedId = 3000;
const emptyReleaseId = 4000;
const fetchedAt = "2026-10-23T12:00:00.000Z";

function credit(
  key: string,
  name: string,
  artistId: number | null,
  role: string,
  trackReference: string | null = null,
): ExplorerCredit {
  return { key, name, artistId, role, trackReference, category: "performance" };
}

// These deterministic API fixtures exercise track scope and catalog navigation;
// their numeric IDs are test data, not live Discogs identifiers.
const kindOfBlue: ExplorerRelease = {
  id: kindOfBlueId,
  title: "Kind of Blue",
  artists: [{ id: 100, name: "Miles Davis" }],
  year: 1959,
  label: "Columbia",
  catalogNumber: "CL 1355",
  country: "US",
  format: "Vinyl, LP",
  coverUrl: null,
  sourceUrl: `https://www.discogs.com/release/${kindOfBlueId}`,
  fetchedAt,
  tracks: [
    {
      id: "a1",
      position: "A1",
      title: "So What",
      duration: "9:22",
      type: "track",
      credits: [
        credit("bill-piano", "Bill Evans", 102, "Piano", "A1, A3 to B2"),
      ],
    },
    {
      id: "a2",
      position: "A2",
      title: "Freddie Freeloader",
      duration: "9:46",
      type: "track",
      credits: [
        credit("wynton-piano", "Wynton Kelly", wyntonId, "Piano", "A2"),
      ],
    },
  ],
  releaseCredits: [
    credit("miles-trumpet", "Miles Davis", 100, "Trumpet"),
    credit("paul-bass", "Paul Chambers", 103, "Bass"),
  ],
  unresolvedCredits: [
    {
      ...credit(
        "unresolved",
        "Unresolved source credit",
        null,
        "Remastering",
        "C7",
      ),
      category: "production",
    },
  ],
};
const kellyBlue: ExplorerRelease = {
  ...kindOfBlue,
  id: kellyBlueId,
  title: "Kelly Blue",
  artists: [{ id: wyntonId, name: "Wynton Kelly" }],
  label: "Riverside",
  catalogNumber: "RLP 12-298",
  sourceUrl: `https://www.discogs.com/release/${kellyBlueId}`,
  tracks: [
    {
      id: "kelly-a1",
      position: "A1",
      title: "Kelly Blue",
      duration: "10:46",
      type: "track",
      credits: [credit("kelly-piano", "Wynton Kelly", wyntonId, "Piano", "A1")],
    },
  ],
  releaseCredits: [],
  unresolvedCredits: [],
};
const undiluted: ExplorerRelease = {
  ...kellyBlue,
  id: undilutedId,
  title: "Undiluted",
  sourceUrl: `https://www.discogs.com/release/${undilutedId}`,
};
const emptyRelease: ExplorerRelease = {
  ...kindOfBlue,
  id: emptyReleaseId,
  tracks: [],
  releaseCredits: [],
  unresolvedCredits: [],
};
const wynton: ExplorerArtist = {
  id: wyntonId,
  name: "Wynton Kelly",
  realName: "Wynton Charles Kelly",
  profile: "A fixture profile for the pianist and his musical connections.",
  imageUrl: null,
  sourceUrl: `https://www.discogs.com/artist/${wyntonId}`,
  groups: [{ id: sextetId, name: "The Miles Davis Sextet", active: false }],
  members: [],
  aliases: [{ id: 203, name: "W. Kelly", active: null }],
  fetchedAt,
};
const sextet: ExplorerArtist = {
  ...wynton,
  id: sextetId,
  name: "The Miles Davis Sextet",
  realName: null,
  sourceUrl: `https://www.discogs.com/artist/${sextetId}`,
  groups: [],
  members: [
    { id: wyntonId, name: "Wynton Kelly", active: false },
    { id: 103, name: "Paul Chambers", active: false },
  ],
  aliases: [],
};

function discography(page: number): ExplorerDiscography {
  return {
    artistId: wyntonId,
    page,
    pages: 2,
    total: 2,
    fetchedAt,
    releases:
      page === 1
        ? [
            {
              id: 800,
              type: "master",
              mainReleaseId: kellyBlueId,
              title: "Kelly Blue",
              artist: "Wynton Kelly",
              year: 1959,
              thumbUrl: null,
              role: "Main",
              sourceUrl: "https://www.discogs.com/master/800",
            },
          ]
        : [
            {
              id: undilutedId,
              type: "release",
              mainReleaseId: null,
              title: "Undiluted",
              artist: "Wynton Kelly",
              year: 1965,
              thumbUrl: null,
              role: "Main",
              sourceUrl: `https://www.discogs.com/release/${undilutedId}`,
            },
          ],
  };
}

function candidate(id: number, label: string): CatalogCandidate {
  return {
    discogsId: id,
    artist: "Miles Davis",
    title: "Kind of Blue",
    year: 1959,
    genres: ["Jazz"],
    label,
    catalogNumber: "CL 1355",
    country: "US",
    format: "Vinyl, LP",
    coverUrl: null,
    sourceUrl: `https://www.discogs.com/release/${id}`,
  };
}

interface ApiState {
  artistRequests: number[];
  releaseRequests: number[];
  pages: number[];
  catalogRequests: string[];
  writes: string[];
  failArtist: boolean;
  failPage: number | null;
  failRelease: number | null;
}

async function setup(page: Page, manual = false) {
  const record = {
    ...DEFAULT_COPY,
    ...candidate(manual ? 0 : kindOfBlueId, "Columbia"),
    discogsId: manual ? null : kindOfBlueId,
    id: copyId,
    releaseId: "00000000-0000-4000-8000-000000000102",
    version: 1,
    addedAt: fetchedAt,
    updatedAt: fetchedAt,
    crateIds: [],
    spinCount: 0,
    lastPlayedAt: null,
    photoIds: [],
  };
  const library: LibraryData = {
    collector: {
      id: "00000000-0000-4000-8000-000000000001",
      name: "Explorer fixture",
      email: "explorer-browser@example.com",
      isOwner: false,
    },
    records: [record],
    crates: [],
    sharedCrates: [],
    listens: [],
  };
  const state: ApiState = {
    artistRequests: [],
    releaseRequests: [],
    pages: [],
    catalogRequests: [],
    writes: [],
    failArtist: false,
    failPage: null,
    failRelease: null,
  };
  await page.route("**/api/library", (route) =>
    route.fulfill({ json: library }),
  );
  await page.route(`**/api/records/${copyId}`, (route) => {
    state.writes.push(route.request().method());
    return route.fulfill({
      status: 405,
      json: { error: "Exploration must not change the copy." },
    });
  });
  await page.route(`**/api/records/${copyId}/explore`, (route) =>
    route.fulfill({
      json: {
        copyId,
        artist: "Miles Davis",
        title: "Kind of Blue",
        discogsId: manual ? null : kindOfBlueId,
        release: manual ? null : kindOfBlue,
      } satisfies ExplorerStart,
    }),
  );
  await page.route("**/api/catalog?**", (route) => {
    state.catalogRequests.push(new URL(route.request().url()).search);
    return route.fulfill({
      json: {
        candidates: [
          candidate(kindOfBlueId, "Columbia original"),
          candidate(emptyReleaseId, "Alternate pressing, sparse credits"),
        ],
      },
    });
  });
  await page.route("**/api/explore/artists/*", (route) => {
    const id = Number(
      new URL(route.request().url()).pathname.split("/").at(-1),
    );
    state.artistRequests.push(id);
    if (state.failArtist) {
      state.failArtist = false;
      return route.fulfill({
        status: 503,
        json: {
          error:
            "Artist information is temporarily unavailable. Please try again.",
        },
      });
    }
    return route.fulfill({ json: id === sextetId ? sextet : wynton });
  });
  await page.route("**/api/explore/artists/*/releases?**", (route) => {
    const pageNumber = Number(
      new URL(route.request().url()).searchParams.get("page"),
    );
    state.pages.push(pageNumber);
    if (state.failPage === pageNumber) {
      state.failPage = null;
      return route.fulfill({
        status: 429,
        json: { error: "The catalog is busy. Give it a moment and try again." },
      });
    }
    return route.fulfill({ json: discography(pageNumber) });
  });
  await page.route("**/api/explore/releases/*", (route) => {
    const id = Number(
      new URL(route.request().url()).pathname.split("/").at(-1),
    );
    state.releaseRequests.push(id);
    if (state.failRelease === id) {
      state.failRelease = null;
      return route.fulfill({
        status: 503,
        json: {
          error:
            "Release information is temporarily unavailable. Please try again.",
        },
      });
    }
    const release =
      id === kellyBlueId
        ? kellyBlue
        : id === undilutedId
          ? undiluted
          : id === emptyReleaseId
            ? emptyRelease
            : kindOfBlue;
    return route.fulfill({ json: release });
  });
  return state;
}

async function selectFreddie(page: Page) {
  await expect(
    page.getByRole("region", { name: "On this song", exact: true }),
  ).toBeVisible();
  const select = page.getByRole("combobox", {
    name: "Choose a song",
    exact: true,
  });
  if (await select.isVisible()) await select.selectOption("a2");
  else {
    const track = page
      .getByRole("list", { name: "Tracks", exact: true })
      .getByRole("button", { name: /Freddie Freeloader/ });
    await track.focus();
    await track.press("Enter");
  }
  await expect(
    page
      .getByRole("region", { name: "On this song", exact: true })
      .getByRole("heading", { name: "Freddie Freeloader" }),
  ).toBeVisible();
}

async function expectNoOverflow(page: Page) {
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth + 1,
    ),
  ).toBe(true);
}

test("song-specific credits lead through an artist tree and albums without losing the selected track", async ({
  page,
}) => {
  const state = await setup(page);
  await page.goto(`/records/${copyId}`);
  await page.getByRole("link", { name: "Artist Tree", exact: false }).click();
  await expect(page).toHaveURL(new RegExp(`/records/${copyId}/explore$`));
  const song = page.getByRole("region", { name: "On this song", exact: true });
  const releaseCredits = page.getByRole("region", {
    name: "Release credits",
    exact: true,
  });
  await expect(
    song.getByRole("heading", { name: "So What", exact: true }),
  ).toBeVisible();
  await expect(
    song.getByRole("button", { name: /Bill Evans.*Piano/ }),
  ).toBeVisible();
  await expect(song.getByRole("button", { name: /Wynton Kelly/ })).toHaveCount(
    0,
  );
  await expect(
    song.getByRole("button", { name: /Paul Chambers|Miles Davis/ }),
  ).toHaveCount(0);
  await expect(
    releaseCredits.getByRole("button", { name: /Paul Chambers.*Bass/ }),
  ).toBeVisible();
  await expect(releaseCredits).toContainText(
    "They aren’t confirmed for the selected song",
  );
  await page
    .getByText("Credits with unresolved track references", { exact: false })
    .click();
  await expect(
    page.getByText("Listed tracks: C7", { exact: true }),
  ).toBeVisible();
  await selectFreddie(page);
  await expect(song.getByRole("button", { name: /Bill Evans/ })).toHaveCount(0);
  if (test.info().project.name === "desktop-chromium") {
    await page
      .locator('[class*="creditsPanel"]')
      .screenshot({ path: ".wax-local/explorer-credits-desktop.png" });
    await page
      .getByRole("region", { name: "Kind of Blue", exact: true })
      .evaluate((element) =>
        element.scrollIntoView({ block: "start", behavior: "instant" }),
      );
    await page.screenshot({
      path: ".wax-local/explorer-studio-viewport-desktop.png",
    });
  } else if (test.info().project.name === "mobile-webkit") {
    await page.locator('[class*="creditsPanel"]').screenshot({
      path: ".wax-local/explorer-credits-mobile.png",
      style: ".mobile-nav, nextjs-portal { visibility: hidden !important; }",
    });
    await page
      .getByRole("region", { name: "Kind of Blue", exact: true })
      .evaluate((element) =>
        element.scrollIntoView({ block: "start", behavior: "instant" }),
      );
    await page.screenshot({
      path: ".wax-local/explorer-studio-viewport-mobile.png",
    });
  }
  await song.getByRole("button", { name: /Wynton Kelly.*Piano/ }).click();
  const tree = page.getByRole("region", { name: /^Artist Tree/ });
  await expect(
    tree.getByRole("heading", { name: "Wynton Kelly", exact: true }),
  ).toBeVisible();
  await expect(tree.getByRole("heading", { name: /Groups/ })).toBeVisible();
  await expect(tree.getByRole("heading", { name: /Members/ })).toBeVisible();
  await expect(
    tree.getByRole("button", { name: "W. Kelly", exact: true }),
  ).toBeVisible();
  expect(state.pages).toEqual([]);
  const group = tree.getByRole("button", { name: /The Miles Davis Sextet/ });
  await group.focus();
  await group.press("Enter");
  await expect(
    tree.getByRole("heading", { name: "The Miles Davis Sextet", exact: true }),
  ).toBeVisible();
  await expect(
    tree.getByRole("button", { name: /Wynton Kelly/ }),
  ).toBeVisible();
  await expect(
    tree.getByRole("button", { name: /Paul Chambers/ }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Back one step", exact: true })
    .click();
  await expect(
    tree.getByRole("heading", { name: "Wynton Kelly", exact: true }),
  ).toBeVisible();
  await expectNoOverflow(page);
  if (test.info().project.name === "desktop-chromium") {
    await tree.screenshot({ path: ".wax-local/explorer-tree-desktop.png" });
    await page.screenshot({
      path: ".wax-local/explorer-studio-desktop.png",
      fullPage: true,
    });
  } else if (test.info().project.name === "mobile-webkit") {
    await tree.screenshot({
      path: ".wax-local/explorer-tree-mobile.png",
      style: ".mobile-nav, nextjs-portal { visibility: hidden !important; }",
    });
    await page.screenshot({
      path: ".wax-local/explorer-studio-mobile.png",
      fullPage: true,
    });
  }
  await tree
    .getByRole("button", { name: "Explore their records", exact: false })
    .click();
  await expect(
    tree.getByText("2 catalog entries · Page 1 of 2", { exact: true }),
  ).toBeVisible();
  await tree.getByRole("button", { name: "Next records", exact: true }).click();
  await expect(
    tree.getByRole("button", { name: "Explore Undiluted", exact: true }),
  ).toBeVisible();
  await tree
    .getByRole("button", { name: "Previous records", exact: true })
    .click();
  await tree
    .getByRole("button", { name: "Explore Kelly Blue", exact: true })
    .click();
  await expect(
    page.getByRole("region", { name: "Kelly Blue", exact: true }),
  ).toBeVisible();
  expect(state.releaseRequests).toEqual([kellyBlueId]);
  expect(state.releaseRequests).not.toContain(800);
  expect(state.pages).toEqual([1, 2, 1]);
  await page
    .getByRole("button", { name: "Back one step", exact: true })
    .click();
  await expect(
    tree.getByRole("heading", { name: "Wynton Kelly", exact: true }),
  ).toBeVisible();
  await expect(
    tree.getByRole("button", { name: "Explore Kelly Blue", exact: true }),
  ).toBeVisible();
  await expect(
    song.getByRole("heading", { name: "Freddie Freeloader", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Back one step", exact: true })
    .click();
  await expect(tree).toHaveCount(0);
  await expect(
    song.getByRole("heading", { name: "Freddie Freeloader", exact: true }),
  ).toBeVisible();
  await expectNoOverflow(page);
  expect(state.writes).toEqual([]);
});

test("catalog outages and throttling preserve the selected song, artist, and album page for retry", async ({
  page,
}) => {
  const state = await setup(page);
  state.failArtist = true;
  await page.goto(`/records/${copyId}/explore`);
  await selectFreddie(page);
  const song = page.getByRole("region", { name: "On this song", exact: true });
  await song.getByRole("button", { name: /Wynton Kelly.*Piano/ }).click();
  await expect(
    page
      .getByRole("alert")
      .filter({ hasText: "Artist information is temporarily unavailable" }),
  ).toBeVisible();
  await expect(
    song.getByRole("heading", { name: "Freddie Freeloader", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Try connection again", exact: true })
    .click();
  const tree = page.getByRole("region", { name: /^Artist Tree/ });
  await expect(
    tree.getByRole("heading", { name: "Wynton Kelly", exact: true }),
  ).toBeVisible();
  await tree
    .getByRole("button", { name: "Explore their records", exact: false })
    .click();
  await expect(
    tree.getByRole("button", { name: "Explore Kelly Blue", exact: true }),
  ).toBeVisible();
  state.failPage = 2;
  await tree.getByRole("button", { name: "Next records", exact: true }).click();
  await expect(tree.getByRole("alert")).toContainText("The catalog is busy");
  await expect(
    tree.getByRole("button", { name: "Explore Kelly Blue", exact: true }),
  ).toBeVisible();
  await expect(
    tree.getByText("2 catalog entries · Page 1 of 2", { exact: true }),
  ).toBeVisible();
  await tree
    .getByRole("button", { name: "Try records again", exact: true })
    .click();
  await expect(
    tree.getByRole("button", { name: "Explore Undiluted", exact: true }),
  ).toBeVisible();
  state.failRelease = undilutedId;
  await tree
    .getByRole("button", { name: "Explore Undiluted", exact: true })
    .click();
  await expect(
    page
      .getByRole("alert")
      .filter({ hasText: "Release information is temporarily unavailable" }),
  ).toBeVisible();
  await expect(
    tree.getByRole("heading", { name: "Wynton Kelly", exact: true }),
  ).toBeVisible();
  await expect(
    tree.getByText("2 catalog entries · Page 2 of 2", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Try connection again", exact: true })
    .click();
  await expect(
    page.getByRole("region", { name: "Undiluted", exact: true }),
  ).toBeVisible();
  expect(state.artistRequests).toEqual([wyntonId, wyntonId]);
  expect(state.pages).toEqual([1, 2, 2]);
  expect(state.releaseRequests).toEqual([undilutedId, undilutedId]);
  expect(state.writes).toEqual([]);
});

test("manual records require an explicit release choice and missing credits never invent players", async ({
  page,
}) => {
  const state = await setup(page, true);
  await page.goto(`/records/${copyId}/explore`);
  await expect(
    page.getByRole("heading", {
      name: "Choose a catalog release to explore.",
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.getByLabel("Artist", { exact: true })).toHaveValue(
    "Miles Davis",
  );
  await expect(page.getByLabel("Release title", { exact: true })).toHaveValue(
    "Kind of Blue",
  );
  expect(state.catalogRequests).toEqual([]);
  expect(state.releaseRequests).toEqual([]);
  await page
    .getByRole("button", { name: "Find releases", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Explore credits", exact: true }),
  ).toHaveCount(2);
  expect(state.releaseRequests).toEqual([]);
  await page
    .getByRole("button", { name: "Explore credits", exact: true })
    .nth(1)
    .click();
  await expect(
    page.getByText("No track list is available for this release.", {
      exact: false,
    }),
  ).toBeVisible();
  await expect(
    page.getByText("No specific song credits listed.", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("No additional release credits are listed.", {
      exact: false,
    }),
  ).toBeVisible();
  const song = page.getByRole("region", { name: "On this song", exact: true });
  await expect(song.getByRole("button")).toHaveCount(0);
  await expect(
    page.getByText("Exploring never changes your copy.", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", {
      name: "Choose another catalog release",
      exact: false,
    }),
  ).toBeVisible();
  expect(state.releaseRequests).toEqual([emptyReleaseId]);
  expect(new URLSearchParams(state.catalogRequests[0]).get("artist")).toBe(
    "Miles Davis",
  );
  expect(new URLSearchParams(state.catalogRequests[0]).get("title")).toBe(
    "Kind of Blue",
  );
  expect(state.writes).toEqual([]);
  await expectNoOverflow(page);
});
