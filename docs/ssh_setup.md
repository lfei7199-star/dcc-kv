# SSH 部署密钥配置指南

> 给 DCC-KV 项目的合作者用。
> 配合 `git_strategy.md` 一起读。

## 为什么用 SSH Deploy Key 而不是 PAT

- **更安全**：key 是 key，不是密码
- **可独立撤销**：删 Deploy Key 立即失效，不影响别的设备
- **可粒度控制**：每个 key 限定到具体 repo + 写权限
- **不用存 token**：token 容易被误 commit；key pair 的 private key 留在本地

## 5 分钟配置流程

### 1. 生成 key pair

```bash
# 替换 mavis-sandbox-dcc-kv 为你的标识
ssh-keygen -t ed25519 \
  -C "your-name-dcc-kv-deploy" \
  -f ~/.ssh/dcc_kv_deploy \
  -N ""   # 无 passphrase
```

输出：
- `~/.ssh/dcc_kv_deploy` — **private key，绝对不能上传**
- `~/.ssh/dcc_kv_deploy.pub` — public key，**可以上传到 GitHub**

### 2. 添加到 GitHub

1. 打开 https://github.com/lfei7199-star/dcc-kv/settings/keys/new
   （把 `lfei7199-star` 换成你的用户名）
2. Title: `mavis-sandbox`（或你的标识）
3. Key: 粘贴 `~/.ssh/dcc_kv_deploy.pub` 的全部内容
4. ☑ **Allow write access**（必勾）
5. 点 **Add key**

### 3. 配置 SSH config（推荐）

`~/.ssh/config`：
```
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/dcc_kv_deploy
  IdentitiesOnly yes
```

**好处**：
- push 不用每次指定 key
- `IdentitiesOnly yes` 防止 SSH 自动尝试别的 key

### 4. 测试连通性

```bash
ssh -T git@github.com
```

成功：
```
Hi lfei7199-star/dcc-kv! You've successfully authenticated, but GitHub does not provide shell access.
```

### 5. Push

```bash
cd /workspace/dcc_kv
git push -u origin main
git push origin v0.2.0-m1
```

## 在 CI 里用 Deploy Key

如果 GitHub Actions 要 clone 你的私人仓库（CI 默认不能）：

1. Settings → Secrets and variables → Actions → New repository secret
2. Name: `DCC_KV_DEPLOY_KEY`
3. Value: `~/.ssh/dcc_kv_deploy` 的**全部内容**（private key）
4. Workflow 里：
   ```yaml
   - name: Setup SSH
     run: |
       mkdir -p ~/.ssh
       echo "${{ secrets.DCC_KV_DEPLOY_KEY }}" > ~/.ssh/deploy_key
       chmod 600 ~/.ssh/deploy_key
       ssh-keyscan github.com >> ~/.ssh/known_hosts
   ```

## 撤销

任何时候：

1. https://github.com/lfei7199-star/dcc-kv/settings/keys
2. 找到 `mavis-sandbox`
3. 点 **Delete**

**立即生效**。你本地的 private key 留着也没用了。

## 多设备 / 多合作者

每个合作者生成自己的 key pair（不同 comment）：
```bash
ssh-keygen -t ed25519 -C "alice-dcc-kv" -f ~/.ssh/dcc_kv_alice -N ""
ssh-keygen -t ed25519 -C "bob-dcc-kv"   -f ~/.ssh/dcc_kv_bob   -N ""
```

每个 key 独立加到 GitHub。撤销某人只删他那把。

## 故障排查

### Permission denied (publickey)

1. 检查 key 是否加到 GitHub：Settings → Deploy keys
2. 检查 local key 路径：`ssh-add -l`
3. 测试连通性：`ssh -vT git@github.com`（verbose）

### Key loaded but still denied

1. 检查 `IdentitiesOnly yes` 是否在 ssh config
2. 检查 Deploy key **Allow write access** 是否勾上
3. 检查仓库 owner / collab 权限

### Lost private key

1. 在 GitHub 删 Deploy key
2. 重新生成 + 重新添加
3. 老的 key 没法用（私钥不在了）

## 已经在沙箱里的 key

我们已经生成了 `/workspace/.ssh/dcc_kv_deploy` + `.pub`，并 push 成功。
- 如果你**继续用这个沙箱**协作：key 留着，没事
- 如果你**只用本机**协作：现在就可以去 GitHub 删这个 Deploy key

## 安全 checklist

- [ ] Private key 文件权限 `600`（不是 644 / 755）
- [ ] Private key **不** commit 到 git
- [ ] `.gitignore` 里有 `*.key` `*.pem` `deploy_key`（默认已加）
- [ ] 用完即删 Deploy key
- [ ] CI 用 fine-grained PAT 而非 Deploy key 的 private key 长期存（折中方案）
