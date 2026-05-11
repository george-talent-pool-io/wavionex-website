import * as argon2 from '@node-rs/argon2';

/* argon2id is the OWASP-recommended password hash. These parameters target
   ~100ms per hash on Vercel Hobby — slow enough to deter brute force, fast
   enough to not blow the 10s function timeout under burst signup load.
   Tune via env at your own risk. */
const OPTS: argon2.Options = {
    algorithm: argon2.Algorithm.Argon2id,
    memoryCost: 19_456, // 19 MiB
    timeCost: 2,
    parallelism: 1
};

export async function hashPassword(plain: string): Promise<string> {
    return argon2.hash(plain, OPTS);
}

export async function verifyPassword(hash: string, plain: string): Promise<boolean> {
    try {
        return await argon2.verify(hash, plain);
    } catch {
        return false;
    }
}
