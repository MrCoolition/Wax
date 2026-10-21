"use client";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useRouter } from "next/navigation";
import type { LibraryData } from "@/lib/contracts";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public details?: unknown,
  ) {
    super(message);
  }
}
export async function api<T>(url: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(url, {
    ...init,
    cache: "no-store",
    headers: {
      ...(init.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...init.headers,
    },
  });
  const text = await response.text();
  let result: unknown;
  try {
    result = text ? JSON.parse(text) : {};
  } catch {
    throw new ApiError(
      "The server could not complete that request. Please try again.",
      response.status,
    );
  }
  if (!response.ok) {
    const error = result as { error?: string; details?: unknown };
    throw new ApiError(
      error.error || "Something went wrong. Please try again.",
      response.status,
      error.details,
    );
  }
  return result as T;
}
const LibraryContext = createContext<{
  data: LibraryData | null;
  loading: boolean;
  error: string;
  refresh: () => Promise<void>;
  search: string;
  setSearch: (value: string) => void;
} | null>(null);
export function LibraryProvider({ children }: { children: ReactNode }) {
  const [data, setData] = useState<LibraryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const router = useRouter();
  const sequence = useRef(0);
  const refresh = useCallback(async () => {
    const current = ++sequence.current;
    try {
      const result = await api<LibraryData>("/api/library");
      if (current === sequence.current) {
        setData(result);
        setError("");
      }
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        router.replace("/sign-in");
        return;
      }
      if (current === sequence.current)
        setError(
          err instanceof Error
            ? err.message
            : "Could not load your collection.",
        );
    } finally {
      if (current === sequence.current) setLoading(false);
    }
  }, [router]);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  return (
    <LibraryContext.Provider
      value={{ data, loading, error, refresh, search, setSearch }}
    >
      {children}
    </LibraryContext.Provider>
  );
}
export function useLibrary() {
  const value = useContext(LibraryContext);
  if (!value) throw new Error("LibraryProvider missing");
  return value;
}
