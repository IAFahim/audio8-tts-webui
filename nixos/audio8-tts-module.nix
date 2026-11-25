# Sanitized NixOS module template for Edge0/Audio8-TTS-Realistic-Air Web UI.
# Import into your host configuration (and ensure NVIDIA CUDA drivers + nix-ld are enabled).
{ config, lib, pkgs, ... }:

let
  # Wrapper injects the venv's NVIDIA wheel libs (libnvrtc, libcudnn, etc.) into LD_LIBRARY_PATH
  # alongside /run/opengl-driver/lib and nix-ld so PyTorch CUDA kernels run cleanly on NixOS.
  audio8Webui = pkgs.writeShellApplication {
    name = "audio8-tts-webui";
    runtimeInputs = [ pkgs.ffmpeg-headless ];
    text = ''
      shopt -s nullglob
      nv_libs=""
      for d in /var/lib/audio8-tts/.venv/lib/python*/site-packages/nvidia/*/lib; do
        nv_libs="''${nv_libs:+''${nv_libs}:}''${d}"
      done
      export LD_LIBRARY_PATH="''${nv_libs:+''${nv_libs}:}/run/opengl-driver/lib:/run/current-system/sw/share/nix-ld/lib"
      export NIX_LD="/run/current-system/sw/share/nix-ld/lib/ld.so"
      export NIX_LD_LIBRARY_PATH="/run/current-system/sw/share/nix-ld/lib"
      export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
      export HF_HUB_OFFLINE="1"
      export TRANSFORMERS_OFFLINE="1"
      export AUDIO8_MODEL_DIR="/var/lib/audio8-tts/model"
      export AUDIO8_DATA_DIR="/var/lib/audio8-tts"
      export AUDIO8_WEBUI_DIR="/var/lib/audio8-tts/webui"
      exec /var/lib/audio8-tts/.venv/bin/python -m uvicorn \
        --app-dir /var/lib/audio8-tts/webui server:app --host 0.0.0.0 --port 8999
    '';
  };
in
{
  users.groups.audio8 = { };
  users.users.audio8 = {
    isSystemUser = true;
    group = "audio8";
    home = "/var/lib/audio8-tts";
    createHome = true;
    extraGroups = [ "video" "render" ];
  };

  systemd.tmpfiles.rules = [
    "d /var/lib/audio8-tts 0755 audio8 audio8 - -"
    "d /var/lib/audio8-tts/voices 0755 audio8 audio8 - -"
    "d /var/lib/audio8-tts/outputs 0755 audio8 audio8 - -"
  ];

  # On-demand service (not auto-started at boot): `sudo systemctl start audio8-tts.service`
  systemd.services.audio8-tts = {
    description = "Audio8-TTS Realistic-Air (0.6B) PyTorch Web UI & API";
    after = [ "network.target" "nvidia-persistenced.service" ];
    wantedBy = [ ];
    path = [ pkgs.ffmpeg-headless ];
    unitConfig = {
      ConditionPathExists = [
        "/var/lib/audio8-tts/model/model.safetensors"
        "/var/lib/audio8-tts/.venv/bin/python"
        "/var/lib/audio8-tts/webui/server.py"
      ];
    };
    environment = {
      HOME = "/var/lib/audio8-tts";
    };
    serviceConfig = {
      Type = "simple";
      User = "audio8";
      Group = "audio8";
      WorkingDirectory = "/var/lib/audio8-tts";
      ExecStart = "${audio8Webui}/bin/audio8-tts-webui";
      Restart = "on-failure";
      RestartSec = 5;
    };
  };

  # Optional: open TCP 8999 on your LAN / Tailscale interface if accessing remotely:
  # networking.firewall.interfaces."eth0".allowedTCPPorts = [ 8999 ];
}
