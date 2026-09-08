FROM node:22-alpine AS build
WORKDIR /app
RUN corepack enable
COPY package.json tsconfig.json ./
COPY src ./src
RUN pnpm install --no-frozen-lockfile && pnpm build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/package.json ./
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
EXPOSE 8787
CMD ["node", "dist/index.js"]
