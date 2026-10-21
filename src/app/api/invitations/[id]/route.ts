import { requireCollector } from "@/server/auth";
import { revokeInvitation } from "@/server/invitations";
import { handleApi } from "@/server/http";
export async function DELETE(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  return handleApi(request, async () =>
    revokeInvitation(await requireCollector(), (await context.params).id),
  );
}
