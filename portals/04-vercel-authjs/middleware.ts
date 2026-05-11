import { auth } from '@/lib/auth';
import { NextResponse } from 'next/server';

export default auth((req) => {
    const url = req.nextUrl;
    if (url.pathname.startsWith('/dashboard') && !req.auth?.user) {
        const u = url.clone();
        u.pathname = '/login';
        return NextResponse.redirect(u);
    }
});

export const config = {
    matcher: ['/dashboard/:path*']
};
