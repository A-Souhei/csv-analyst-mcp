# sourced by start.sh and update-volumes.sh — not executable on its own

m_host=() m_dest=()

# Writes docker-compose.override.yml and fills m_host/m_dest with the dataset mounts.
generate_override() {
  m_host=() m_dest=()
  {
    echo "# generated from .env — do not edit"
    echo "services:"
    echo "  csv-analyst:"
    echo "    ports:"
    echo "      - \"${PORT:-41733}:41733\""
    # "utility" injects nvidia-smi and libnvidia-ml only — no CUDA runtime — which is
    # all llm_status needs to read VRAM. Skipped entirely on hosts without the runtime.
    if docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
      echo "    runtime: nvidia"
      echo "    environment:"
      echo "      NVIDIA_VISIBLE_DEVICES: all"
      echo "      NVIDIA_DRIVER_CAPABILITIES: utility"
    fi
    if [ -n "${CSV_SOURCES:-}" ]; then
      echo "    volumes:"
      IFS=',' read -ra entries <<<"$CSV_SOURCES"
      for e in "${entries[@]}"; do
        e="${e#"${e%%[![:space:]]*}"}"          # trim leading space
        # "path:name" — the name is optional, and without it the mount is named
        # after the directory (not the whole host path, which would nest under /mnt)
        if [[ "$e" == *:* ]]; then
          host="${e%:*}" name="${e##*:}"
        else
          host="$e" name="$(basename "$e")"
        fi
        name="${name//\//_}"
        [ -n "$name" ] || name="$(basename "$host")"
        [ -d "$host" ] || { echo "warning: skipping missing dir: $host" >&2; continue; }
        echo "      - ${host}:/mnt/${name}:ro"
        m_host+=("$host") m_dest+=("/mnt/${name}")
      done
    fi
  } > docker-compose.override.yml
}

# A bind mount pins the source directory's inode when the container starts, so a
# host dir that is replaced rather than edited in place (re-clone, atomic mv) keeps
# serving the old contents forever. Compose sees unchanged config and won't recreate,
# so compare inodes across the mount and force it when they diverge.
stale=()
detect_stale() {
  stale=()
  local cid host_ino cont_ino i
  cid="$(docker compose ps -q csv-analyst 2>/dev/null || true)"
  [ -n "$cid" ] || return 0
  for i in "${!m_host[@]}"; do
    host_ino="$(stat -c %i "${m_host[$i]}" 2>/dev/null)" || continue
    cont_ino="$(docker exec "$cid" stat -c %i "${m_dest[$i]}" 2>/dev/null)" || continue
    [ "$host_ino" = "$cont_ino" ] || stale+=("${m_dest[$i]}")
  done
}
