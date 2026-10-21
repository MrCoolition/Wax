import { z } from "zod";
import { requireCollector } from "@/server/auth";
import { handleApi } from "@/server/http";
import { decideProposal } from "@/server/wizard";

export const runtime = "nodejs";
export async function POST(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  return handleApi(request, async () => {
    const collector = await requireCollector();
    const { action } = z
      .object({ action: z.enum(["approve", "reject"]) })
      .strict()
      .parse(await request.json());
    return decideProposal(collector.id, (await context.params).id, action);
  });
}
