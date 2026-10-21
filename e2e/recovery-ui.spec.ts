import { test, expect } from "@playwright/test";

const token = "owner-recovery-browser-fixture-1234567890";
const password = "Recovery browser fixture 123!";

test("recovery preserves entered passwords and does not repeat a completed reset", async ({
  page,
}) => {
  let resets = 0;
  let signIns = 0;
  await page.route("**/api/recovery**", async (route) => {
    if (new URL(route.request().url()).pathname === "/api/recovery/inspect") {
      expect(route.request().method()).toBe("POST");
      expect(route.request().postDataJSON()).toEqual({ token });
      await route.fulfill({ json: { email: "owner@example.com" } });
      return;
    }
    resets++;
    if (resets === 1) {
      await route.fulfill({
        status: 503,
        json: { error: "Please try again." },
      });
      return;
    }
    expect(route.request().postDataJSON()).toEqual({ token, password });
    await route.fulfill({ json: { ok: true } });
  });
  await page.route("**/api/auth/sign-in/email", async (route) => {
    signIns++;
    await route.fulfill({
      status: 503,
      json: {
        message: "Sign-in is temporarily unavailable.",
        code: "UNAVAILABLE",
      },
    });
  });

  await page.goto("/recover#token=" + token);
  await expect(page.getByLabel("Email address")).toHaveValue(
    "owner@example.com",
  );
  if (test.info().project.name === "desktop-chromium")
    await page.screenshot({
      path: ".wax-local/recovery-preview.png",
      fullPage: true,
    });
  await expect(page.getByLabel("Email address")).toHaveAttribute(
    "autocomplete",
    "username",
  );
  const nextPassword = page
    .getByLabel("New password", { exact: false })
    .first();
  const confirmation = page.getByLabel("Confirm new password");
  await nextPassword.fill(password);
  await confirmation.fill("A different password 123!");
  await page.getByRole("button", { name: "Update my password" }).click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "passwords do not match",
  );
  expect(resets).toBe(0);

  await confirmation.fill(password);
  await page
    .getByRole("button", { name: "Show password", exact: true })
    .click();
  await expect(nextPassword).toHaveAttribute("type", "text");
  await page.getByRole("button", { name: "Update my password" }).click();
  await expect(page.getByRole("main").getByRole("alert")).toHaveText(
    "Please try again.",
  );
  await expect(nextPassword).toHaveValue(password);
  await expect(confirmation).toHaveValue(password);
  await page.getByRole("button", { name: "Update my password" }).click();

  await expect(page.getByRole("status")).toContainText(
    "Your password is updated",
  );
  await expect(page).toHaveURL(/\/recover$/);
  await expect(page.getByLabel("Password", { exact: true })).toHaveValue(
    password,
  );
  await expect(page.getByLabel("Password", { exact: true })).toHaveAttribute(
    "autocomplete",
    "current-password",
  );
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.getByRole("main").getByRole("alert")).toHaveText(
    "Sign-in is temporarily unavailable.",
  );
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect.poll(() => signIns).toBe(2);
  expect(resets).toBe(2);
});

test("an invalid recovery link cannot submit a password", async ({ page }) => {
  await page.route("**/api/recovery**", (route) =>
    route.fulfill({
      status: 404,
      json: { error: "This recovery link has expired or already been used." },
    }),
  );
  await page.goto("/recover#token=" + token);
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "expired or already been used",
  );
  await expect(
    page.getByRole("button", { name: "Update my password" }),
  ).toBeDisabled();
});
