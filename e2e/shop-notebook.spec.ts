import { expect, test, type Page } from "@playwright/test";
import type { LibraryData } from "../src/lib/contracts";
import type { NearbyShops, RecordShop } from "../src/lib/shops";
import type { SavedShop, SaveShopInput } from "../src/lib/shop-notebook";

const library: LibraryData = {
  collector: {
    id: "00000000-0000-4000-8000-000000000001",
    name: "WAX browser fixture",
    email: "shop-notebook-browser@example.com",
    isOwner: false,
  },
  records: [],
  crates: [],
  sharedCrates: [],
  listens: [],
};

const shop: RecordShop = {
  id: "node/101",
  name: "The Soul Crate",
  latitude: 40.71642,
  longitude: -74.00218,
  distanceKm: 0.7,
  address: "123 Record Collector Avenue, New York, NY 10013",
  website: "https://example.com/record-shop",
  phone: null,
  openingHours: null,
  osmUrl: "https://www.openstreetmap.org/node/101",
};

interface NotebookMock {
  shops: SavedShop[];
  writes: SaveShopInput[];
  deletes: { shopId: string; version: number }[];
  failNextSave: boolean;
}

async function setup(page: Page, initial: SavedShop[] = []) {
  const notebook: NotebookMock = {
    shops: structuredClone(initial),
    writes: [],
    deletes: [],
    failNextSave: false,
  };
  await page.addInitScript(() => {
    const state = window as typeof window & { locationRequests: number };
    state.locationRequests = 0;
    navigator.geolocation.getCurrentPosition = () => {
      state.locationRequests++;
    };
  });
  await page.route("**/api/library", (route) =>
    route.fulfill({ json: library }),
  );
  await page.route("**/api/shops/places", (route) =>
    route.fulfill({
      json: {
        places: [
          {
            id: "new-york",
            label: "New York, United States",
            latitude: 40.71,
            longitude: -74.01,
          },
        ],
      },
    }),
  );
  await page.route("**/api/shops/nearby", (route) =>
    route.fulfill({
      json: {
        center: { latitude: 40.71, longitude: -74.01 },
        radiusKm: 10,
        fetchedAt: "2026-10-23T12:00:00.000Z",
        limited: false,
        shops: [shop],
      } satisfies NearbyShops,
    }),
  );
  await page.route("**/api/shops/notebook", async (route) => {
    const request = route.request();
    if (request.method() === "GET") {
      await route.fulfill({ json: { shops: notebook.shops } });
      return;
    }
    if (request.method() === "PUT") {
      const input = request.postDataJSON() as SaveShopInput;
      notebook.writes.push(input);
      if (notebook.failNextSave) {
        notebook.failNextSave = false;
        await route.fulfill({
          status: 503,
          json: {
            error:
              "Your notebook is temporarily unavailable. Please try again.",
          },
        });
        return;
      }
      const existing = notebook.shops.find(
        (saved) => saved.shop.id === input.shop.id,
      );
      if ((existing?.version ?? null) !== input.version) {
        await route.fulfill({
          status: 409,
          json: {
            error:
              "This shop changed in another tab. Reload its saved details before editing again.",
          },
        });
        return;
      }
      const saved: SavedShop = {
        shop: input.shop,
        rating: input.rating,
        employees: input.employees,
        version: (existing?.version ?? 0) + 1,
        updatedAt: "2026-10-23T12:00:00.000Z",
      };
      notebook.shops = [
        saved,
        ...notebook.shops.filter((entry) => entry.shop.id !== saved.shop.id),
      ];
      await route.fulfill({ json: saved });
      return;
    }
    if (request.method() === "DELETE") {
      const input = request.postDataJSON() as {
        shopId: string;
        version: number;
      };
      notebook.deletes.push(input);
      notebook.shops = notebook.shops.filter(
        (entry) => entry.shop.id !== input.shopId,
      );
      await route.fulfill({ json: { ok: true } });
      return;
    }
    throw new Error(`Unexpected notebook method: ${request.method()}`);
  });
  return notebook;
}

async function discoverShop(page: Page) {
  await page.goto("/shops");
  await page.getByLabel("Or enter a city or postal code").fill("New York");
  await page.getByRole("button", { name: "Find area" }).click();
  await page
    .getByRole("button", { name: "New York, United States", exact: true })
    .click();
  await expect(page.locator(".shops-results .shop-card h3")).toHaveText([
    shop.name,
  ]);
}

async function expectNoOverflow(page: Page) {
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth + 1,
    ),
  ).toBeTruthy();
}

test("ratings and shop people survive reload, can be edited, and can be removed", async ({
  page,
}) => {
  const notebook = await setup(page);
  await discoverShop(page);
  await page.getByRole("button", { name: "Rate & add people" }).click();
  const editor = page.getByRole("dialog", { name: shop.name, exact: true });
  await expect(editor).toBeVisible();
  await editor.getByRole("radio", { name: "5 stars", exact: true }).check();
  await editor.getByRole("button", { name: "Add a person" }).click();
  const firstPerson = editor.getByRole("group", {
    name: "Person 1",
    exact: true,
  });
  await firstPerson.getByLabel("Name", { exact: true }).fill("Nina");
  await firstPerson
    .getByLabel("Role", { exact: true })
    .fill("Soul & jazz buyer");
  await firstPerson
    .getByRole("radio", { name: "Cosmic", exact: true })
    .locator("..")
    .click();
  await editor.getByRole("button", { name: "Add a person" }).click();
  const secondPerson = editor.getByRole("group", {
    name: "Person 2",
    exact: true,
  });
  await secondPerson.getByLabel("Name", { exact: true }).fill("Sam");
  await secondPerson.getByLabel("Role", { exact: true }).fill("Owner");
  await secondPerson
    .getByRole("radio", { name: "Sunshine", exact: true })
    .locator("..")
    .click();
  await expectNoOverflow(page);
  if (test.info().project.name === "desktop-chromium") {
    await editor.evaluate((element) => element.scrollTo(0, 0));
    await editor.screenshot({
      path: ".wax-local/shop-notebook-editor-desktop.png",
    });
  } else if (test.info().project.name === "mobile-webkit") {
    await editor.evaluate((element) => element.scrollTo(0, 0));
    await editor.screenshot({
      path: ".wax-local/shop-notebook-editor-mobile.png",
    });
    await firstPerson
      .getByRole("group", { name: "Pick their character" })
      .screenshot({
        path: ".wax-local/shop-notebook-avatars-mobile.png",
      });
  }
  await editor
    .getByRole("button", { name: "Save changes", exact: true })
    .click();
  await expect(editor).not.toBeVisible();
  expect(notebook.shops).toHaveLength(1);
  expect(notebook.shops[0]).toMatchObject({
    shop: { id: shop.id, name: shop.name },
    rating: 5,
    version: 1,
    employees: [
      { name: "Nina", role: "Soul & jazz buyer", avatar: "cosmic" },
      { name: "Sam", role: "Owner", avatar: "sunshine" },
    ],
  });
  expect(notebook.shops[0].employees[0].id).not.toBe(
    notebook.shops[0].employees[1].id,
  );

  await page.reload();
  const saved = page.getByRole("region", { name: /^Your shops/ });
  await expect(saved.getByRole("heading", { name: shop.name })).toBeVisible();
  await expect(saved.locator(".shop-people-preview")).toContainText(
    "Nina & Sam",
  );
  await expect(saved.locator(".saved-shop-roles")).toContainText(
    "Nina · Soul & jazz buyer",
  );
  await expect(saved.locator(".saved-shop-roles")).toContainText("Sam · Owner");
  await expect(
    saved.getByLabel("Your rating: 5 out of 5 stars", { exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () =>
        (window as typeof window & { locationRequests: number })
          .locationRequests,
    ),
  ).toBe(0);
  await expect(page.locator(".shops-results")).toHaveCount(0);
  await expectNoOverflow(page);
  if (test.info().project.name === "desktop-chromium") {
    await saved.screenshot({
      path: ".wax-local/shop-notebook-desktop.png",
    });
  } else if (test.info().project.name === "mobile-webkit") {
    await saved.screenshot({
      path: ".wax-local/shop-notebook-mobile.png",
    });
  }

  await saved.getByRole("button", { name: "Edit shop notes" }).click();
  await expect(
    firstPerson.getByRole("radio", { name: "Cosmic", exact: true }),
  ).toBeChecked();
  await expect(
    secondPerson.getByRole("radio", { name: "Sunshine", exact: true }),
  ).toBeChecked();
  await editor.getByRole("button", { name: "Clear rating" }).click();
  await firstPerson
    .getByRole("button", { name: "Remove person 1", exact: true })
    .click();
  await editor
    .getByRole("button", { name: "Save changes", exact: true })
    .click();
  await expect(editor).not.toBeVisible();
  expect(notebook.shops[0]).toMatchObject({ rating: null, version: 2 });
  expect(notebook.shops[0].employees).toHaveLength(1);
  expect(notebook.shops[0].employees[0].name).toBe("Sam");
  await expect(saved.locator(".saved-shop-roles")).not.toContainText("Nina");
  await expect(saved.locator(".shop-people-preview")).toHaveText("Sam");

  await saved.getByRole("button", { name: "Edit shop notes" }).click();
  await editor
    .getByRole("button", { name: "Remove saved shop", exact: true })
    .click();
  await page.getByRole("button", { name: "Remove shop", exact: true }).click();
  await expect(editor).not.toBeVisible();
  expect(notebook.deletes).toEqual([{ shopId: shop.id, version: 2 }]);
  await page.reload();
  await expect(saved.getByRole("heading", { name: shop.name })).toHaveCount(0);
});

test("a failed save keeps rating, people, and avatar choices for a retry", async ({
  page,
}) => {
  const notebook = await setup(page);
  notebook.failNextSave = true;
  await discoverShop(page);
  await page.getByRole("button", { name: "Rate & add people" }).click();
  const editor = page.getByRole("dialog", { name: shop.name, exact: true });
  await editor.getByRole("radio", { name: "4 stars", exact: true }).check();
  await editor.getByRole("button", { name: "Add a person" }).click();
  const person = editor.getByRole("group", { name: "Person 1", exact: true });
  await person.getByLabel("Name", { exact: true }).fill("Nina");
  await person.getByLabel("Role", { exact: true }).fill("Soul & jazz buyer");
  await person
    .getByRole("radio", { name: "Cosmic", exact: true })
    .press("Space");
  await editor
    .getByRole("button", { name: "Save changes", exact: true })
    .click();
  await expect(editor.getByRole("alert")).toContainText(
    "temporarily unavailable",
  );
  expect(notebook.shops).toHaveLength(0);
  await expect(
    editor.getByRole("radio", { name: "4 stars", exact: true }),
  ).toBeChecked();
  await expect(person.getByLabel("Name", { exact: true })).toHaveValue("Nina");
  await expect(person.getByLabel("Role", { exact: true })).toHaveValue(
    "Soul & jazz buyer",
  );
  await expect(
    person.getByRole("radio", { name: "Cosmic", exact: true }),
  ).toBeChecked();
  await editor.getByRole("button", { name: "Cancel", exact: true }).click();
  await page.getByRole("button", { name: "Keep editing", exact: true }).click();
  await expect(person.getByLabel("Name", { exact: true })).toHaveValue("Nina");
  await editor
    .getByRole("button", { name: "Save changes", exact: true })
    .click();
  await expect(editor).not.toBeVisible();
  expect(notebook.writes).toHaveLength(2);
  expect(notebook.writes[0]).toEqual(notebook.writes[1]);
  expect(notebook.shops[0]).toMatchObject({
    rating: 4,
    employees: [{ avatar: "cosmic" }],
  });
});

test("a stale edit preserves its draft until the collector chooses to load the latest notes", async ({
  page,
}) => {
  const notebook = await setup(page, [
    {
      shop: {
        id: shop.id,
        name: shop.name,
        latitude: shop.latitude,
        longitude: shop.longitude,
        address: shop.address,
        website: shop.website,
      },
      rating: 3,
      employees: [
        {
          id: "00000000-0000-4000-8000-000000000002",
          name: "Nina",
          role: "Buyer",
          avatar: "cosmic",
        },
      ],
      version: 1,
      updatedAt: "2026-10-23T12:00:00.000Z",
    },
  ]);
  await page.goto("/shops");
  const saved = page.getByRole("region", { name: /^Your shops/ });
  await saved.getByRole("button", { name: "Edit shop notes" }).click();
  const editor = page.getByRole("dialog", { name: shop.name, exact: true });
  const person = editor.getByRole("group", { name: "Person 1", exact: true });
  await person.getByLabel("Role", { exact: true }).fill("My unsaved role");
  await editor.getByRole("radio", { name: "5 stars", exact: true }).check();
  // A second tab has saved a newer version since this editor opened.
  notebook.shops[0] = {
    ...notebook.shops[0],
    version: 2,
    rating: 4,
    employees: [
      { ...notebook.shops[0].employees[0], role: "Head buyer", avatar: "mint" },
    ],
  };
  await editor
    .getByRole("button", { name: "Save changes", exact: true })
    .click();
  await expect(editor.getByRole("alert")).toContainText(
    "changed in another tab",
  );
  await expect(person.getByLabel("Role", { exact: true })).toHaveValue(
    "My unsaved role",
  );
  await expect(
    editor.getByRole("radio", { name: "5 stars", exact: true }),
  ).toBeChecked();
  await expect(
    editor.getByRole("button", { name: "Save changes", exact: true }),
  ).toBeDisabled();
  expect(notebook.shops[0].employees[0].role).toBe("Head buyer");
  await editor
    .getByRole("button", { name: "Load latest notes", exact: true })
    .click();
  await expect(
    editor.getByText(
      "Replace your unsaved changes with the latest saved notes?",
      { exact: true },
    ),
  ).toBeVisible();
  await expect(person.getByLabel("Role", { exact: true })).toHaveValue(
    "My unsaved role",
  );
  await editor
    .getByRole("button", { name: "Replace with latest", exact: true })
    .click();
  await expect(person.getByLabel("Role", { exact: true })).toHaveValue(
    "Head buyer",
  );
  await expect(
    person.getByRole("radio", { name: "Mint", exact: true }),
  ).toBeChecked();
  await expect(
    editor.getByRole("radio", { name: "4 stars", exact: true }),
  ).toBeChecked();
  await person
    .getByLabel("Role", { exact: true })
    .fill("Head buyer / jazz expert");
  await editor
    .getByRole("button", { name: "Save changes", exact: true })
    .click();
  await expect(editor).not.toBeVisible();
  expect(notebook.shops[0]).toMatchObject({
    version: 3,
    rating: 4,
    employees: [{ role: "Head buyer / jazz expert", avatar: "mint" }],
  });
});
