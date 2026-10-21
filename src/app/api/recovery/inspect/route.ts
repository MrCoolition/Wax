import { z } from "zod";
import { limitRegistration } from "@/server/invitations";
import { handleApi } from "@/server/http";
import { inspectRecovery } from "@/server/recovery";

export async function POST(request: Request) {
  return handleApi(request, async () => {
    await limitRegistration(request);
    const input = z
      .object({ token: z.string() })
      .strict()
      .parse(await request.json());
    return inspectRecovery(input.token);
  });
}
