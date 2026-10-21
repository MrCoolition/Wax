import { requireCollector } from "@/server/auth";
import { handleApi, HttpError } from "@/server/http";
import { askWizard, wizardInputSchema } from "@/server/wizard";

export const runtime = "nodejs";
export const maxDuration = 90;
export async function POST(request: Request) {
  return handleApi(request, async () => {
    const collector = await requireCollector();
    if (Number(request.headers.get("content-length")) > 100_000)
      throw new HttpError(
        413,
        "This conversation is too long. Start a new question.",
      );
    return askWizard(
      collector.id,
      wizardInputSchema.parse(await request.json()),
    );
  });
}
