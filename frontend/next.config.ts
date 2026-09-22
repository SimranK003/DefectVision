import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone", // smaller production Docker image - see infra/docker/frontend.Dockerfile
};

export default nextConfig;
