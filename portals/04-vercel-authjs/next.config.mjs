/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    /* The argon2 packages contain native code; mark them external so Next.js
       doesn't try to bundle them through Webpack/Turbopack. */
    serverExternalPackages: ['@node-rs/argon2', '@neondatabase/serverless']
};
export default nextConfig;
