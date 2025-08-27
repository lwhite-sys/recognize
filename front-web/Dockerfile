# 构建阶段
FROM node:20-alpine3.19 AS builder

# 设置 npm 镜像源
RUN npm config set registry https://registry.npmmirror.com/

# 安装 pnpm 并设置 pnpm 镜像源
RUN npm install -g pnpm@latest && pnpm config set registry https://registry.npmmirror.com/
# 将前端源码复制到 /blog-zhjs 目录中
WORKDIR /blog-zhjs

COPY packages /blog-zhjs/packages
# 复制整个项目，除了 node_modules 和 dist 等目录
COPY  package.json pnpm-lock.yaml pnpm-workspace.yaml /blog-zhjs/

# 使用 pnpm 安装依赖
RUN pnpm install --frozen-lockfile

COPY . .

# 编译项目
RUN pnpm run build

# 部署阶段
FROM nginx:1.19.6

# 复制 nginx.conf 配置文件到镜像中
COPY ["./_nginx/default.conf", "/etc/nginx/nginx.conf"]

# 删除 nginx html 原有内容
RUN rm -rf /usr/share/nginx/html/*

EXPOSE 80

# 从编译镜像复制编译结果到此镜像
COPY --from=builder /blog-zhjs/dist /usr/share/nginx/html