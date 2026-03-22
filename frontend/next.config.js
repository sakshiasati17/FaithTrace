/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  experimental: {
    serverActions: {
      allowedOrigins: ["localhost:3000", "0.0.0.0:3000"],
    },
  },
};

module.exports = nextConfig;
