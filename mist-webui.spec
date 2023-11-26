# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import copy_metadata
datas = []

datas += copy_metadata('absl-py')
datas += copy_metadata('accelerate')
datas += copy_metadata('aiofiles')
datas += copy_metadata('aiohttp')
datas += copy_metadata('aiosignal')
datas += copy_metadata('altair')
datas += copy_metadata('altgraph')
datas += copy_metadata('annotated-types')
datas += copy_metadata('anyio')
datas += copy_metadata('appdirs')
datas += copy_metadata('async-timeout')
datas += copy_metadata('attrs')
datas += copy_metadata('cache')
datas += copy_metadata('cachetools')
datas += copy_metadata('cffi')
datas += copy_metadata('click')
datas += copy_metadata('contourpy')
datas += copy_metadata('cycler')
datas += copy_metadata('datasets')
datas += copy_metadata('diffusers')
datas += copy_metadata('dill')
datas += copy_metadata('docker-pycreds')
datas += copy_metadata('exceptiongroup')
datas += copy_metadata('fastapi')
datas += copy_metadata('ffmpy')
datas += copy_metadata('fire')
datas += copy_metadata('flatbuffers')
datas += copy_metadata('fonttools')
datas += copy_metadata('frozenlist')
datas += copy_metadata('ftfy')
datas += copy_metadata('gitdb')
datas += copy_metadata('gitpython')
datas += copy_metadata('google-auth')
datas += copy_metadata('google-auth-oauthlib')
datas += copy_metadata('gradio')
datas += copy_metadata('gradio-client')
datas += copy_metadata('grpcio')
datas += copy_metadata('h11')
datas += copy_metadata('httpcore')
datas += copy_metadata('httpx')
datas += copy_metadata('importlib-resources')
datas += copy_metadata('jinja2')
datas += copy_metadata('jsonschema')
datas += copy_metadata('jsonschema-specifications')
datas += copy_metadata('kiwisolver')
datas += copy_metadata('markdown')
datas += copy_metadata('markdown-it-py')
datas += copy_metadata('markupsafe')
datas += copy_metadata('matplotlib')
datas += copy_metadata('mdurl')
datas += copy_metadata('mediapipe')
datas += copy_metadata('mpmath')
datas += copy_metadata('multidict')
datas += copy_metadata('multiprocess')
datas += copy_metadata('networkx')
datas += copy_metadata('numpy')
datas += copy_metadata('oauthlib')
datas += copy_metadata('opencv-contrib-python')
datas += copy_metadata('opencv-python')
datas += copy_metadata('orjson')
datas += copy_metadata('pandas')
datas += copy_metadata('pathtools')
datas += copy_metadata('pefile')
datas += copy_metadata('protobuf')
datas += copy_metadata('psutil')
datas += copy_metadata('pyarrow')
datas += copy_metadata('pyasn1')
datas += copy_metadata('pyasn1-modules')
datas += copy_metadata('pycparser')
datas += copy_metadata('pydantic')
datas += copy_metadata('pydantic-core')
datas += copy_metadata('pydub')
datas += copy_metadata('pygments')
datas += copy_metadata('pyinstaller')
datas += copy_metadata('pyinstaller-hooks-contrib')
datas += copy_metadata('pynvml')
datas += copy_metadata('pyparsing')
datas += copy_metadata('python-dateutil')
datas += copy_metadata('python-multipart')
datas += copy_metadata('pytz')
datas += copy_metadata('pywin32-ctypes')
datas += copy_metadata('referencing')
datas += copy_metadata('requests-oauthlib')
datas += copy_metadata('responses')
datas += copy_metadata('rich')
datas += copy_metadata('rpds-py')
datas += copy_metadata('rsa')
datas += copy_metadata('safetensors')
datas += copy_metadata('scipy')
datas += copy_metadata('semantic-version')
datas += copy_metadata('sentry-sdk')
datas += copy_metadata('setproctitle')
datas += copy_metadata('shellingham')
datas += copy_metadata('six')
datas += copy_metadata('smmap')
datas += copy_metadata('sniffio')
datas += copy_metadata('sounddevice')
datas += copy_metadata('starlette')
datas += copy_metadata('sympy')
datas += copy_metadata('tensorboard')
datas += copy_metadata('tensorboard-data-server')
datas += copy_metadata('termcolor')
datas += copy_metadata('tokenizers')
datas += copy_metadata('tomlkit')
datas += copy_metadata('toolz')
datas += copy_metadata('torch')
datas += copy_metadata('torchvision')
datas += copy_metadata('transformers')
datas += copy_metadata('typer')
datas += copy_metadata('typing-extensions')
datas += copy_metadata('tzdata')
datas += copy_metadata('urllib3')
datas += copy_metadata('uvicorn')
datas += copy_metadata('wandb')
datas += copy_metadata('wcwidth')
datas += copy_metadata('websockets')
datas += copy_metadata('werkzeug')
datas += copy_metadata('xformers')
datas += copy_metadata('xxhash')
datas += copy_metadata('yarl')

a = Analysis(
    ['mist-webui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['attacks', 'attacks.ita_webui', 'attacks.utils', 'lora_diffusion', 'lora_diffusion.lora', 'lora_diffusion.xformers_utils'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='mist-webui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='mist-webui',
)
