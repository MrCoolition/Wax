import { requireCollector } from "@/server/auth";
import { listInvitations, createInvitation } from "@/server/invitations";
import { handleApi } from "@/server/http";
export async function GET(request: Request) {
  return handleApi(request, async () =>
    listInvitations(await requireCollector()),
  );
}
export async function POST(request: Request) {
  return handleApi(request, async () => {
    const user = await requireCollector();
    return createInvitation(user, (await request.json()).email);
  });
}
