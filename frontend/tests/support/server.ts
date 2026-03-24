import { spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import net from "node:net";
import os from "node:os";
import path from "node:path";

export type E2EServer = {
  baseURL: string;
  stop: () => Promise<void>;
  workspace: string;
};

type SpawnedProcess = ReturnType<typeof spawn>;

function getFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        server.close(() => reject(new Error("Could not find a free port.")));
        return;
      }

      const port = address.port;
      server.close(() => resolve(port));
    });
  });
}

async function waitForServer(url: string): Promise<void> {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        return;
      }
    } catch {
      // Keep polling until the server is listening.
    }

    await new Promise((resolve) => setTimeout(resolve, 200));
  }

  throw new Error(`Timed out waiting for ${url}`);
}

function waitForExit(process: SpawnedProcess): Promise<void> {
  return new Promise((resolve) => {
    if (process.exitCode !== null || process.signalCode !== null) {
      resolve();
      return;
    }

    process.once("exit", () => resolve());
  });
}

export async function startE2EServer(): Promise<E2EServer> {
  const workspace = await mkdtemp(
    path.join(os.tmpdir(), "image-yolo-faces-e2e-"),
  );
  const port = await getFreePort();
  const script = path.join(
    process.cwd(),
    "frontend/tests/support/e2e_server.py",
  );

  const child = spawn(
    "uv",
    ["run", "python", script, "--workspace", workspace, "--port", String(port)],
    {
      cwd: process.cwd(),
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  let stderr = "";
  child.stdout.on("data", (chunk) => {
    process.stdout.write(chunk);
  });
  child.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
    process.stderr.write(chunk);
  });

  const baseURL = `http://127.0.0.1:${port}`;
  try {
    await waitForServer(`${baseURL}/`);
  } catch (error) {
    child.kill("SIGTERM");
    await waitForExit(child);
    await rm(workspace, { recursive: true, force: true });
    const message =
      stderr.trim().length > 0
        ? `${String(error)}\n\nServer stderr:\n${stderr}`
        : String(error);
    throw new Error(message);
  }

  return {
    baseURL,
    workspace,
    stop: async () => {
      child.kill("SIGTERM");
      await Promise.race([
        waitForExit(child),
        new Promise((resolve) => setTimeout(resolve, 5_000)),
      ]);
      child.kill("SIGKILL");
      await waitForExit(child);
      await rm(workspace, { recursive: true, force: true });
    },
  };
}
