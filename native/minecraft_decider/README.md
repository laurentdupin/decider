# Decider Minecraft client

This Fabric 26.3 client mod runs Decider 4B through the persistent InferBridge
JSONL host. Minecraft supplies compact structured state; Decider selects one of
seven movement macros; the vanilla client applies the corresponding key state.
It does not use screenshots or generative output.

The initial milestone is intentionally narrow: safe local exploration only.
It can walk, steer, turn, jump, or stop. It cannot mine, attack, craft, use
items, or join servers automatically. It releases all controls whenever a GUI
or pause screen is open and forces `stop` below six health.

The earlier command-macro task is retained only as historical prototype code
and is disabled at runtime. Current policy permits only player-equivalent
controls: movement, camera rotation, attack/use, hotbar selection, and normal
inventory or container clicks. World and server access must remain read-only.

## Build

```powershell
native\minecraft_decider\gradlew.bat -p native\minecraft_decider build
```

The project targets Minecraft 26.3, Fabric Loader 0.19.5, Fabric API
0.161.0+26.3, Baritone 1.20.0, Loom 1.17, and Java 25. Baritone is used only
through its API as a player-input pathfinder and miner; its chat commands and
builder/world-edit interfaces are not used.

Download the official `baritone-api-fabric-1.20.0.jar` release into
`out/dependencies` before building. Its expected SHA-256 is
`49adfc063cfbfd0b6f08e9d814359807baa2d1768c0d39d6c5968268547cbca6`.

## Local configuration

Install the remapped mod JAR and Fabric API in `.minecraft\mods`. Create
`.minecraft\config\decider-minecraft.json` with absolute paths to
`decider_native_jsonl.exe` and the Decider 4B GGUF. The runner parameters use
a 2,048-token context and set `disable_vision=true`.

The mod is disarmed unless this file exists:

```text
.minecraft\config\decider-autopilot.enabled
```

The legacy marker below is rejected and automatically removed:

```text
.minecraft\config\decider-house-chicken.enabled
```

The survival player-action controller uses this additional marker:

```text
.minecraft\config\decider-survival.enabled
```

Its first verified milestone lets Decider choose a wood target, then delegates
walking, aiming, and block breaking to Baritone's player-control API. Completion
is based solely on logs actually present in the player's inventory.

Create that empty marker only after entering a disposable local Creative
world. Delete or rename it to disarm the controller. Opening any Minecraft GUI
also releases movement immediately. Decisions and probabilities are recorded
under `.minecraft\logs\latest.log` with logger name `decider-minecraft`.

On the validated three-GPU machine, Minecraft renders on the RX 9070 while the
configuration pins Decider 4B to Vulkan device 2, the RX 6700 XT. This avoids
placing the 8.41 GB BF16 model on the game's rendering GPU.
