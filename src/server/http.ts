import { ZodError } from "zod";
export class HttpError extends Error {
  constructor(
    public status: number,
    message: string,
    public details?: unknown,
  ) {
    super(message);
    this.name = "HttpError";
  }
}
export async function handleApi(
  request: Request,
  action: () => Promise<unknown>,
): Promise<Response> {
  try {
    if (!["GET", "HEAD", "OPTIONS"].includes(request.method.toUpperCase())) {
      const origin = request.headers.get("origin");
      const allowed = new Set([new URL(request.url).origin]);
      if (process.env.BETTER_AUTH_URL)
        allowed.add(new URL(process.env.BETTER_AUTH_URL).origin);
      if (!origin || !allowed.has(origin))
        throw new HttpError(403, "This request must come from WAX.");
    }
    const result = await action();
    if (result instanceof Response) return result;
    return Response.json(result ?? { ok: true }, {
      headers: { "Cache-Control": "private, no-store" },
    });
  } catch (error) {
    if (error instanceof HttpError)
      return Response.json(
        {
          error: error.message,
          ...(error.details === undefined ? {} : { details: error.details }),
        },
        { status: error.status, headers: { "Cache-Control": "no-store" } },
      );
    if (error instanceof ZodError)
      return Response.json(
        {
          error: "Please check the highlighted information.",
          details: error.flatten(),
        },
        { status: 400 },
      );
    if (error instanceof SyntaxError)
      return Response.json(
        { error: "The request could not be read." },
        { status: 400 },
      );
    const code = (error as { code?: string })?.code;
    if (code === "23505")
      return Response.json(
        { error: "That entry already exists." },
        { status: 409 },
      );
    if (code === "23503" || code === "22P02" || code === "23514")
      return Response.json(
        { error: "That record or selection is not valid." },
        { status: 400 },
      );
    console.error("WAX request failed", {
      name: error instanceof Error ? error.name : "UnknownError",
      code,
    });
    return Response.json(
      { error: "WAX could not finish this request. Please try again." },
      { status: 500 },
    );
  }
}
