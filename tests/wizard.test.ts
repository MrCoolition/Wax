import { describe, expect, it, vi } from "vitest";
vi.mock("../src/server/db", () => ({
  getPool: vi.fn(),
  withTransaction: vi.fn(),
}));
vi.mock("../src/server/collection", () => ({
  getLibrary: vi.fn(),
  approveChanges: vi.fn(),
}));
import { validateWizardReply } from "../src/server/wizard";

const owned = "23bca1b7-2dea-40ee-9949-4b323e9247a5";
const other = "791a265b-1987-422f-aa08-3960fb5d3825";
const fields = {
  notes: null,
  rating: null,
  location: null,
  intent: null,
  mediaCondition: null,
  sleeveCondition: null,
};
const proposal = (
  id = owned,
  version = 3,
  changes: Record<string, unknown> = { ...fields, location: "Shelf A" },
) => ({
  reply: "Review this change.",
  recordIds: [owned],
  proposal: {
    summary: "Move to Shelf A",
    changes: [{ recordId: id, version, changes }],
  },
});

describe("Wizard output authorization boundary", () => {
  it("accepts actual copy IDs and strips unused nullable fields", () => {
    expect(
      validateWizardReply(proposal(), [{ id: owned, version: 3 }]).proposal
        ?.changes,
    ).toEqual([
      { recordId: owned, version: 3, changes: { location: "Shelf A" } },
    ]);
  });
  it("rejects another collector’s copy or a version the model invented", () => {
    expect(() =>
      validateWizardReply(proposal(other), [{ id: owned, version: 3 }]),
    ).toThrow();
    expect(() =>
      validateWizardReply(proposal(owned, 4), [{ id: owned, version: 3 }]),
    ).toThrow();
    expect(() =>
      validateWizardReply(
        { reply: "Play this", recordIds: [other], proposal: null },
        [{ id: owned, version: 3 }],
      ),
    ).toThrow();
  });
  it("rejects unauthorized properties and empty edits", () => {
    const unsafe = proposal();
    Object.assign(unsafe.proposal.changes[0].changes, { purchasePrice: "0" });
    expect(() =>
      validateWizardReply(unsafe, [{ id: owned, version: 3 }]),
    ).toThrow();
    expect(() =>
      validateWizardReply(proposal(owned, 3, fields), [
        { id: owned, version: 3 },
      ]),
    ).toThrow();
  });
  it("leaves a read-only answer without any proposal", () => {
    expect(
      validateWizardReply(
        {
          reply: "Try this overlooked record.",
          recordIds: [owned, owned],
          proposal: null,
        },
        [{ id: owned, version: 3 }],
      ),
    ).toEqual({
      reply: "Try this overlooked record.",
      recordIds: [owned],
      proposal: null,
    });
  });
});
