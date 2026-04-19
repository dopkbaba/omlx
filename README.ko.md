<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/icon-rounded-dark.svg" width="140">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/icon-rounded-light.svg" width="140">
    <img alt="oMLX" src="docs/images/icon-rounded-light.svg" width="140">
  </picture>
</p>

<h1 align="center">oMLX</h1>
<p align="center"><b>Mac에 최적화된 LLM 추론 서버</b><br>Continuous Batching과 다단계 KV 캐시로 최적화된 추론 서버를, 메뉴바에서 편리하게</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <a href="https://buymeacoffee.com/jundot"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

<p align="center">
  <a href="mailto:junkim.dot@gmail.com">junkim.dot@gmail.com</a> · <a href="https://omlx.ai/me">https://omlx.ai/me</a>
</p>

<p align="center">
  <a href="#설치">설치</a> ·
  <a href="#빠른-시작">빠른 시작</a> ·
  <a href="#기능">기능</a> ·
  <a href="#모델">모델</a> ·
  <a href="#cli-설정">CLI 설정</a> ·
  <a href="https://omlx.ai/benchmarks">벤치마크</a> ·
  <a href="https://omlx.ai">oMLX.ai</a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh.md">中文</a> ·
  <b>한국어</b> ·
  <a href="README.ja.md">日本語</a>
</p>

---

<p align="center">
  <img src="docs/images/omlx_dashboard.png" alt="oMLX 관리자 대시보드" width="800">
</p>

> *맥에서 LLM서버 많이 써보셨나요? 다들 많은 불편함을 겪으셨을 겁니다. 어떤 서버는 터미널에서 온갖 명령어를 입력해야 해서 불편하고, 어떤 서버는 GUI인데 이게 편리한가? 싶을만큼 너무 복잡하죠. 저는 그냥 자주 쓰는 모델을 고정 로딩하고, 무거운 모델은 필요할 때 자동으로 교체하고, 컨텍스트 제한을 설정하고, 모든 걸 그냥 쉽고 빠르게 사용하고 싶었습니다.*
>
> *oMLX는 Hot Cache-메모리와 Cold Cache-SSD의 두 계층에 걸쳐 KV캐시를 유지합니다. 대화 중간에 컨텍스트가 바뀌어도 모든 과거 컨텍스트가 캐시에 남습니다. Claude Code 같은 도구로 실제 코딩 작업을 해보셨나요? oMLX와 함께라면 가능합니다.*

## 설치

### macOS 앱

[Releases](https://github.com/jundot/omlx/releases)에서 `.dmg`를 다운로드하고, Applications에 드래그하면 끝입니다. 앱 내 자동 업데이트를 지원하므로 이후 업그레이드는 클릭 한 번이면 됩니다. macOS 앱은 `omlx` CLI 명령어를 포함하지 않습니다. 터미널 사용이 필요하면 Homebrew 또는 소스에서 설치하세요.

### Homebrew

```bash
brew tap jundot/omlx https://github.com/jundot/omlx
brew install omlx

# 최신 버전으로 업그레이드
brew update && brew upgrade omlx

# 백그라운드 서비스로 실행 (크래시 시 자동 재시작)
brew services start omlx

# 선택사항: MCP (Model Context Protocol) 지원
/opt/homebrew/opt/omlx/libexec/bin/pip install mcp
```

### 소스에서 설치

```bash
git clone https://github.com/jundot/omlx.git
cd omlx
pip install -e .          # 코어만
pip install -e ".[mcp]"   # MCP (Model Context Protocol) 포함
```

Python 3.10+와 Apple Silicon (M1/M2/M3/M4)이 필요합니다.

> **개인 메모:** M2 Max (96GB) 환경에서 테스트함. 소스 설치 시 `pip install -e .` 전에 `pip install --upgrade pip` 를 먼저 실행하는 것을 권장.

## 빠른 시작

### macOS 앱

Applications 폴더에서 oMLX를 실행하세요. 환영 화면에서 세 단계만 따라하면 됩니다 — 모델 디렉토리 설정, 서버 시작, 첫 모델 다운로드. 끝입니다.

<p align="center">
  <img src="docs/images/Screenshot 2026-02-10 at 00.36.32.png" alt="oMLX 환영 화면" width="360">
  <img src="docs/images/Screenshot 2026-02-10 at 00.34.30.pn
