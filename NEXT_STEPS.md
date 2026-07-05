# 重新登录后执行

## 1. 验证 docker 权限

```bash
docker ps
```
能跑就说明 docker 组生效了。

## 2. 推送镜像到 GitHub Container Registry

```bash
docker tag fpp-golden-window ghcr.io/gis-blackcaat/fpp-golden-window:latest
docker push ghcr.io/gis-blackcaat/fpp-golden-window:latest
```

## 3. 推送成功后，任何人可以一键运行

```bash
docker run -it ghcr.io/gis-blackcaat/fpp-golden-window:latest demo
```

## 4. 后续更新镜像（代码有改动后）

```bash
cd ~/桌面/release/github
git pull
docker build -t fpp-golden-window -f docker/Dockerfile .
docker tag fpp-golden-window ghcr.io/gis-blackcaat/fpp-golden-window:latest
docker push ghcr.io/gis-blackcaat/fpp-golden-window:latest
```
