package ai.decider.minecraft;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.network.chat.Component;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.level.block.state.BlockState;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

public final class DeciderMinecraftClient implements ClientModInitializer {
    private static final Logger LOGGER = LoggerFactory.getLogger("decider-minecraft");
    private static final Gson GSON = new Gson();
    private static final int DECISION_INTERVAL_TICKS = 10;
    private static final int ACTION_DURATION_TICKS = 10;

    private final ExecutorService inference = Executors.newSingleThreadExecutor(runnable -> {
        Thread thread = new Thread(runnable, "decider-minecraft-inference");
        thread.setDaemon(true);
        return thread;
    });
    private final AtomicBoolean requestPending = new AtomicBoolean();
    private final AtomicReference<Decision> nextDecision = new AtomicReference<>();
    private final AtomicReference<String> failure = new AtomicReference<>();
    private Runner runner;
    private int tick;
    private int actionTicks;
    private String activeAction = "stop";
    private boolean announced;
    private long retryAfterNanos;

    @Override
    public void onInitializeClient() {
        ClientTickEvents.END_CLIENT_TICK.register(this::onTick);
        LOGGER.info("Decider Minecraft client initialized; create config/decider-autopilot.enabled to arm it");
    }

    private void onTick(Minecraft client) {
        LocalPlayer player = client.player;
        if (player == null || client.level == null || client.gui.screen() != null || !isArmed()) {
            releaseControls(client);
            announced = false;
            return;
        }
        if (!announced) {
            player.sendSystemMessage(Component.literal(
                "Decider 4B autopilot armed: movement-only exploration"));
            LOGGER.info("Decider 4B autopilot armed: movement-only exploration");
            announced = true;
        }

        String currentFailure = failure.getAndSet(null);
        if (currentFailure != null) {
            player.sendSystemMessage(Component.literal(
                "Decider error: " + currentFailure));
            LOGGER.error("Decider inference failed: {}", currentFailure);
        }

        Decision decision = nextDecision.getAndSet(null);
        if (decision != null) {
            activeAction = decision.action();
            actionTicks = ACTION_DURATION_TICKS;
            LOGGER.info(String.format(Locale.ROOT,
                "Decider: %s (p=%.3f, %.0f ms)", decision.action(),
                decision.probability(), decision.latency().toNanos() / 1_000_000.0));
        }

        releaseControls(client);
        if (player.getHealth() <= 6.0F) {
            activeAction = "stop";
            actionTicks = 0;
        } else if (actionTicks-- > 0) {
            applyAction(client, player, activeAction);
        }

        if (++tick % DECISION_INTERVAL_TICKS == 0 &&
            System.nanoTime() >= retryAfterNanos &&
            requestPending.compareAndSet(false, true)) {
            JsonObject request = buildRequest(client, player, activeAction);
            inference.submit(() -> decide(request));
        }
    }

    private static boolean isArmed() {
        return Files.isRegularFile(FabricLoader.getInstance().getConfigDir()
            .resolve("decider-autopilot.enabled"));
    }

    private void decide(JsonObject request) {
        long started = System.nanoTime();
        try {
            if (runner == null) runner = Runner.start(loadConfiguration());
            JsonObject response = runner.exchange(GSON.toJson(request));
            JsonObject answer = response.getAsJsonObject("answers").getAsJsonObject("motion");
            String action = answer.get("choice").getAsString();
            double probability = answer.getAsJsonObject("probabilities").get(action).getAsDouble();
            nextDecision.set(new Decision(action, probability,
                Duration.ofNanos(System.nanoTime() - started)));
        } catch (Exception exception) {
            failure.set(exception.getClass().getSimpleName() + ": " + exception.getMessage());
            retryAfterNanos = System.nanoTime() + Duration.ofSeconds(10).toNanos();
            if (runner != null) runner.close();
            runner = null;
        } finally {
            requestPending.set(false);
        }
    }

    private static JsonObject buildRequest(Minecraft client, LocalPlayer player,
                                           String previousAction) {
        JsonObject state = new JsonObject();
        JsonObject pose = new JsonObject();
        pose.addProperty("x", round(player.getX()));
        pose.addProperty("y", round(player.getY()));
        pose.addProperty("z", round(player.getZ()));
        pose.addProperty("yaw_degrees", Math.round(player.getYRot()));
        pose.addProperty("on_ground", player.onGround());
        pose.addProperty("in_water", player.isInWater());
        pose.addProperty("health", player.getHealth());
        pose.addProperty("hunger", player.getFoodData().getFoodLevel());
        pose.addProperty("velocity_x", round(player.getDeltaMovement().x));
        pose.addProperty("velocity_y", round(player.getDeltaMovement().y));
        pose.addProperty("velocity_z", round(player.getDeltaMovement().z));
        state.add("player", pose);

        Direction facing = Direction.fromYRot(player.getYRot());
        BlockPos feet = player.blockPosition();
        JsonObject terrain = new JsonObject();
        terrain.addProperty("facing", facing.getName());
        terrain.addProperty("at_feet", blockName(client, feet));
        terrain.addProperty("ahead_feet", blockName(client, feet.relative(facing)));
        terrain.addProperty("ahead_head", blockName(client, feet.relative(facing).above()));
        terrain.addProperty("ahead_floor", blockName(client, feet.relative(facing).below()));
        terrain.addProperty("left_feet", blockName(client, feet.relative(facing.getCounterClockWise())));
        terrain.addProperty("right_feet", blockName(client, feet.relative(facing.getClockWise())));
        state.add("terrain", terrain);

        JsonArray nearby = new JsonArray();
        List<Entity> entities = client.level.getEntities(
            player, player.getBoundingBox().inflate(8.0));
        for (Entity entity : entities.stream().limit(8).toList()) {
            JsonObject item = new JsonObject();
            item.addProperty("type", entity.getType().toString());
            item.addProperty("distance", round(player.distanceTo(entity)));
            item.addProperty("relative_x", round(entity.getX() - player.getX()));
            item.addProperty("relative_z", round(entity.getZ() - player.getZ()));
            nearby.add(item);
        }
        state.add("nearby_entities", nearby);
        state.addProperty("previous_action", previousAction);
        state.addProperty("goal", "Explore safely and keep moving without taking damage.");

        JsonObject criteria = new JsonObject();
        criteria.addProperty("forward", "Walk straight ahead when the path and floor are clear");
        criteria.addProperty("forward_left", "Move forward while steering left around an obstacle");
        criteria.addProperty("forward_right", "Move forward while steering right around an obstacle");
        criteria.addProperty("jump_forward", "Jump forward over a one-block obstacle or small gap");
        criteria.addProperty("turn_left", "Turn left in place to search for a safer route");
        criteria.addProperty("turn_right", "Turn right in place to search for a safer route");
        criteria.addProperty("stop", "Stop because moving is unsafe or no route is supported by the state");
        JsonObject question = new JsonObject();
        question.addProperty("type", "choice");
        question.addProperty("instructions",
            "Choose the safest useful movement for the next half-second. Avoid falls, liquids, collisions, and hostile entities.");
        question.add("criteria", criteria);
        JsonObject questions = new JsonObject();
        questions.add("motion", question);
        JsonObject request = new JsonObject();
        request.add("state", state);
        request.add("questions", questions);
        request.addProperty("independent", true);
        return request;
    }

    private static String blockName(Minecraft client, BlockPos position) {
        BlockState state = client.level.getBlockState(position);
        return state.isAir() ? "air" : state.getBlock().getName().getString();
    }

    private static double round(double value) {
        return Math.rint(value * 10.0) / 10.0;
    }

    private static void releaseControls(Minecraft client) {
        client.options.keyUp.setDown(false);
        client.options.keyLeft.setDown(false);
        client.options.keyRight.setDown(false);
        client.options.keyJump.setDown(false);
    }

    private static void applyAction(Minecraft client, LocalPlayer player, String action) {
        switch (action) {
            case "forward" -> client.options.keyUp.setDown(true);
            case "forward_left" -> {
                client.options.keyUp.setDown(true);
                client.options.keyLeft.setDown(true);
            }
            case "forward_right" -> {
                client.options.keyUp.setDown(true);
                client.options.keyRight.setDown(true);
            }
            case "jump_forward" -> {
                client.options.keyUp.setDown(true);
                client.options.keyJump.setDown(true);
            }
            case "turn_left" -> player.setYRot(player.getYRot() - 35.0F);
            case "turn_right" -> player.setYRot(player.getYRot() + 35.0F);
            default -> { }
        }
    }

    private static Configuration loadConfiguration() throws IOException {
        Path path = FabricLoader.getInstance().getConfigDir().resolve("decider-minecraft.json");
        try (var reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            return GSON.fromJson(reader, Configuration.class);
        }
    }

    private record Configuration(String executable, String model, String backend,
                                 String gpu, int contextSize, int batchSize,
                                 int threads) { }

    private record Decision(String action, double probability, Duration latency) { }

    private static final class Runner implements AutoCloseable {
        private final Process process;
        private final BufferedWriter input;
        private final BufferedReader output;

        private Runner(Process process) {
            this.process = process;
            this.input = new BufferedWriter(new OutputStreamWriter(
                process.getOutputStream(), StandardCharsets.UTF_8));
            this.output = new BufferedReader(new InputStreamReader(
                process.getInputStream(), StandardCharsets.UTF_8));
        }

        static Runner start(Configuration configuration) throws IOException {
            JsonObject parameters = new JsonObject();
            parameters.addProperty("context_size", configuration.contextSize());
            parameters.addProperty("batch_size", configuration.batchSize());
            parameters.addProperty("threads", configuration.threads());
            parameters.addProperty("disable_vision", true);
            Path parametersPath = FabricLoader.getInstance().getConfigDir()
                .resolve("decider-minecraft-model-parameters.json");
            Files.writeString(parametersPath, GSON.toJson(parameters), StandardCharsets.UTF_8);
            ProcessBuilder builder = new ProcessBuilder(
                configuration.executable(), configuration.model(),
                configuration.backend(), "--model-parameters-file",
                parametersPath.toString(), "--ready");
            builder.environment().put("GGML_VK_VISIBLE_DEVICES", configuration.gpu());
            builder.directory(FabricLoader.getInstance().getGameDir().toFile());
            builder.redirectErrorStream(true);
            Runner runner = new Runner(builder.start());
            StringBuilder diagnostics = new StringBuilder();
            for (int lineCount = 0; lineCount < 64; lineCount++) {
                String line = runner.output.readLine();
                if ("{\"ready\":true}".equals(line)) return runner;
                if (line == null) break;
                if (!line.isBlank()) {
                    if (!diagnostics.isEmpty()) diagnostics.append(" | ");
                    diagnostics.append(line);
                }
            }
            runner.close();
            throw new IOException("native runner exited before ready" +
                (diagnostics.isEmpty() ? "" : ": " + diagnostics));
        }

        synchronized JsonObject exchange(String request) throws IOException {
            input.write(request);
            input.newLine();
            input.flush();
            String response = output.readLine();
            if (response == null) throw new IOException("native runner exited");
            return JsonParser.parseString(response).getAsJsonObject();
        }

        @Override
        public void close() {
            try { input.close(); } catch (IOException ignored) { }
            process.destroy();
        }
    }
}
