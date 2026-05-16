FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json ./

RUN npm ci --omit=dev

COPY index.js ./

ENV PORT=8082

EXPOSE 8082

CMD ["node", "index.js"]
