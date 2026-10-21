import { betterAuth } from "better-auth";
import { APIError } from "better-auth/api";
import { drizzleAdapter } from "@better-auth/drizzle-adapter";
import { headers } from "next/headers";
import type { Collector } from "@/lib/contracts";
import { getDb, getPool } from "./db";
import * as schema from "./schema";
import { getCollector } from "./collection";
import { HttpError } from "./http";

export const normalizeEmail = (email: string) => email.trim().toLowerCase();
export function authBaseUrl(): string {
  const value =
    process.env.BETTER_AUTH_URL ||
    (process.env.VERCEL_URL
      ? `https://${process.env.VERCEL_URL}`
      : "http://localhost:3000");
  return new URL(value).origin;
}
export function createWaxAuth(
  database: Parameters<typeof drizzleAdapter>[0],
  registrationEmail?: string,
) {
  const secret = process.env.BETTER_AUTH_SECRET;
  if (!secret || secret.length < 32)
    throw new Error("Authentication is not configured.");
  return betterAuth({
    appName: "WAX",
    baseURL: authBaseUrl(),
    secret,
    database: drizzleAdapter(database, {
      provider: "pg",
      schema,
      transaction: false,
    }),
    emailAndPassword: {
      enabled: true,
      disableSignUp: !registrationEmail,
      autoSignIn: false,
      requireEmailVerification: false,
      minPasswordLength: 12,
      maxPasswordLength: 128,
    },
    disabledPaths: [
      "/sign-up/email",
      "/request-password-reset",
      "/reset-password",
      "/change-email",
      "/delete-user",
    ],
    trustedOrigins: [authBaseUrl()],
    session: {
      expiresIn: 60 * 60 * 24 * 7,
      updateAge: 60 * 60 * 24,
      cookieCache: { enabled: false },
    },
    rateLimit: {
      enabled: true,
      storage: "database",
      window: 60,
      max: 60,
      customRules: { "/sign-in/email": { window: 60, max: 8 } },
    },
    advanced: {
      cookiePrefix: "wax",
      useSecureCookies: process.env.NODE_ENV === "production",
      ipAddress: { ipAddressHeaders: ["x-real-ip"] },
    },
    databaseHooks: {
      user: {
        create: {
          before: async (user) => {
            if (
              !registrationEmail ||
              normalizeEmail(user.email) !== registrationEmail
            )
              throw new APIError("FORBIDDEN", {
                message: "An invitation is required.",
              });
            return {
              data: { ...user, email: registrationEmail, emailVerified: false },
            };
          },
        },
      },
      session: {
        create: {
          before: async (session) => {
            const collector = (
              await getPool().query(
                "SELECT id FROM wax.collectors WHERE id=$1",
                [session.userId],
              )
            ).rows[0];
            if (!collector)
              throw new APIError("FORBIDDEN", {
                message: "An invitation is required.",
              });
            return { data: session };
          },
        },
      },
    },
    logger: { disabled: true },
    telemetry: { enabled: false },
  });
}
let auth: ReturnType<typeof createWaxAuth> | undefined;
export function getAuth() {
  return (auth ??= createWaxAuth(getDb()));
}
export async function requireCollector(): Promise<Collector> {
  const session = await getAuth().api.getSession({ headers: await headers() });
  if (!session) throw new HttpError(401, "Sign in to your collection.");
  return getCollector(session.user.id);
}
