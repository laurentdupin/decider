package ai.decider.minecraft;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.Minecraft;
import net.minecraft.client.server.IntegratedServer;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.ResourceKey;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.entity.AbstractFurnaceBlockEntity;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.BlockStateProperties;
import net.minecraft.world.level.block.state.properties.DoubleBlockHalf;
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
import java.util.UUID;
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
    private TaskPhase taskPhase = TaskPhase.CHOOSE_MATERIAL;
    private BlockPos taskOrigin;
    private String houseMaterial = "minecraft:oak_planks";
    private BlockPos furnacePosition;
    private int chickenQuantity = 1;
    private boolean awaitingVerification;
    private int verificationTicks;
    private IntegratedServer taskServer;
    private ResourceKey<Level> taskDimension;
    private UUID taskPlayerId;
    private final AtomicBoolean furnacePollPending = new AtomicBoolean();
    private final AtomicReference<FurnaceSnapshot> furnaceSnapshot = new AtomicReference<>();
    private final AtomicReference<Boolean> transferResult = new AtomicReference<>();
    private volatile long sessionGeneration;
    private boolean wasArmed;

    @Override
    public void onInitializeClient() {
        ClientTickEvents.END_CLIENT_TICK.register(this::onTick);
        LOGGER.info("Decider Minecraft client initialized; create config/decider-autopilot.enabled to arm it");
    }

    private void onTick(Minecraft client) {
        LocalPlayer player = client.player;
        boolean armed = isArmed();
        if (!armed) {
            if (wasArmed) endSession();
            releaseControls(client);
            announced = false;
            return;
        }
        if (!wasArmed) {
            wasArmed = true;
            sessionGeneration++;
            if (isHouseChickenTask()) resetTaskState();
        }
        if (player == null || client.level == null || client.gui.screen() != null) {
            releaseControls(client);
            announced = false;
            return;
        }
        if (isHouseChickenTask()) {
            onHouseChickenTick(client, player);
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
            long generation = sessionGeneration;
            inference.submit(() -> decide(request, "motion", generation));
        }
    }

    private void endSession() {
        wasArmed = false;
        sessionGeneration++;
        nextDecision.set(null);
        awaitingVerification = false;
        inference.submit(() -> {
            if (runner != null) runner.close();
            runner = null;
        });
    }

    private void resetTaskState() {
        taskPhase = TaskPhase.CHOOSE_MATERIAL;
        taskOrigin = null;
        furnacePosition = null;
        houseMaterial = "minecraft:oak_planks";
        chickenQuantity = 1;
        awaitingVerification = false;
        verificationTicks = 0;
        taskServer = null;
        taskDimension = null;
        taskPlayerId = null;
        furnaceSnapshot.set(null);
        transferResult.set(null);
    }

    private static boolean isArmed() {
        return Files.isRegularFile(FabricLoader.getInstance().getConfigDir()
            .resolve("decider-autopilot.enabled"));
    }

    private static boolean isHouseChickenTask() {
        return Files.isRegularFile(FabricLoader.getInstance().getConfigDir()
            .resolve("decider-house-chicken.enabled"));
    }

    private void decide(JsonObject request, String questionName, long generation) {
        long started = System.nanoTime();
        try {
            if (runner == null) runner = Runner.start(loadConfiguration());
            JsonObject response = runner.exchange(GSON.toJson(request));
            JsonObject answer = response.getAsJsonObject("answers").getAsJsonObject(questionName);
            String action = answer.get("choice").getAsString();
            double probability = answer.getAsJsonObject("probabilities").get(action).getAsDouble();
            if (generation == sessionGeneration) {
                nextDecision.set(new Decision(action, probability,
                    Duration.ofNanos(System.nanoTime() - started), generation));
            }
        } catch (Exception exception) {
            failure.set(exception.getClass().getSimpleName() + ": " + exception.getMessage());
            retryAfterNanos = System.nanoTime() + Duration.ofSeconds(10).toNanos();
            if (runner != null) runner.close();
            runner = null;
        } finally {
            requestPending.set(false);
        }
    }

    private void onHouseChickenTick(Minecraft client, LocalPlayer player) {
        releaseControls(client);
        if (!client.hasSingleplayerServer() || !player.isCreative() ||
            !client.level.dimension().equals(Level.OVERWORLD)) {
            failAndDisarm(player,
                "House task requires a local Creative world in the Overworld");
            return;
        }
        if (!bindOrValidateTaskSession(client, player)) return;
        if (!announced) {
            player.sendSystemMessage(Component.literal(
                "Decider 4B task armed: build a house and cook chicken"));
            LOGGER.info("Decider house-and-chicken task armed");
            announced = true;
        }
        if (taskOrigin == null) {
            BlockPos feet = player.blockPosition();
            taskOrigin = new BlockPos(feet.getX() + 4, feet.getY() - 1, feet.getZ() - 3);
        }

        String currentFailure = failure.getAndSet(null);
        if (currentFailure != null) {
            player.sendSystemMessage(Component.literal("Decider error: " + currentFailure));
            LOGGER.error("Decider inference failed: {}", currentFailure);
        }

        Decision decision = nextDecision.getAndSet(null);
        if (decision != null && decision.generation() == sessionGeneration &&
            !awaitingVerification) {
            LOGGER.info(String.format(Locale.ROOT,
                "Decider task: %s (p=%.3f, %.0f ms)", decision.action(),
                decision.probability(), decision.latency().toNanos() / 1_000_000.0));
            applyTaskAction(client, player, decision.action());
        }

        if (awaitingVerification) {
            verifyTaskPostcondition(client, player);
            return;
        }
        if (taskPhase == TaskPhase.WAIT_FOR_COOKING) {
            requestFurnaceSnapshot();
            FurnaceSnapshot snapshot = furnaceSnapshot.get();
            if (snapshot != null && snapshot.cookedChicken() > 0)
                taskPhase = TaskPhase.COLLECT_CHICKEN;
            return;
        }
        if (taskPhase == TaskPhase.COMPLETE) return;

        if (++tick % 20 == 0 && System.nanoTime() >= retryAfterNanos &&
            requestPending.compareAndSet(false, true)) {
            JsonObject request = buildTaskRequest(player);
            long generation = sessionGeneration;
            inference.submit(() -> decide(request, "task", generation));
        }
    }

    private JsonObject buildTaskRequest(LocalPlayer player) {
        JsonObject state = new JsonObject();
        state.addProperty("goal", "Build a small safe house, then cook one raw chicken in its furnace.");
        state.addProperty("phase", taskPhase.name().toLowerCase(Locale.ROOT));
        state.addProperty("house_material", houseMaterial);
        state.addProperty("player_x", round(player.getX()));
        state.addProperty("player_y", round(player.getY()));
        state.addProperty("player_z", round(player.getZ()));

        JsonObject criteria = new JsonObject();
        switch (taskPhase) {
            case CHOOSE_MATERIAL -> {
                criteria.addProperty("use_oak", "Use warm oak planks for the house");
                criteria.addProperty("use_cobblestone", "Use durable cobblestone for the house");
                criteria.addProperty("stop", "Do not begin the requested task");
            }
            case PREPARE_SITE -> {
                criteria.addProperty("prepare_site", "Clear and level the selected site before building");
                criteria.addProperty("stop", "Wait without preparing the site");
            }
            case BUILD_SHELL -> {
                criteria.addProperty("build_shell", "Build the floor, walls, and roof now that the site is ready");
                criteria.addProperty("stop", "Wait without constructing the shell");
            }
            case ADD_DETAILS -> {
                criteria.addProperty("add_door_and_windows", "Add a usable door and windows to the completed shell");
                criteria.addProperty("stop", "Leave the shell unfinished");
            }
            case PLACE_FURNACE -> {
                criteria.addProperty("place_furnace", "Place a furnace safely inside the completed house");
                criteria.addProperty("stop", "Wait without placing the furnace");
            }
            case LOAD_CHICKEN -> {
                criteria.addProperty("cook_one_chicken", "Cook one chicken quickly for an immediate meal");
                criteria.addProperty("cook_four_chickens", "Cook four chickens as a larger food supply");
            }
            case LOAD_FUEL -> {
                criteria.addProperty("fuel_with_coal", "Use efficient coal to start cooking now");
                criteria.addProperty("fuel_with_charcoal", "Use renewable charcoal to start cooking now");
            }
            case COLLECT_CHICKEN -> {
                criteria.addProperty("collect_now", "Collect the cooked chicken into inventory now");
                criteria.addProperty("collect_and_finish", "Collect the cooked chicken and finish the task");
            }
            default -> criteria.addProperty("stop", "No further action is required");
        }
        JsonObject question = new JsonObject();
        question.addProperty("type", "choice");
        question.addProperty("instructions",
            "Choose the action that makes safe, direct progress on the stated goal in the current phase.");
        question.add("criteria", criteria);
        JsonObject questions = new JsonObject();
        questions.add("task", question);
        JsonObject request = new JsonObject();
        request.add("state", state);
        request.add("questions", questions);
        request.addProperty("independent", true);
        return request;
    }

    private void applyTaskAction(Minecraft client, LocalPlayer player, String action) {
        switch (taskPhase) {
            case CHOOSE_MATERIAL -> {
                if (action.equals("use_oak")) {
                    houseMaterial = "minecraft:oak_planks";
                    taskPhase = TaskPhase.PREPARE_SITE;
                } else if (action.equals("use_cobblestone")) {
                    houseMaterial = "minecraft:cobblestone";
                    taskPhase = TaskPhase.PREPARE_SITE;
                }
            }
            case PREPARE_SITE -> {
                if (!action.equals("prepare_site")) return;
                command(player, "fill " + box(0, 0, 0, 6, 5, 6) + " minecraft:air");
                command(player, "fill " + box(0, 0, 0, 6, 0, 6) + " " + houseMaterial);
                beginVerification();
            }
            case BUILD_SHELL -> {
                if (!action.equals("build_shell")) return;
                command(player, "fill " + box(0, 1, 0, 6, 3, 0) + " " + houseMaterial);
                command(player, "fill " + box(0, 1, 6, 6, 3, 6) + " " + houseMaterial);
                command(player, "fill " + box(0, 1, 1, 0, 3, 5) + " " + houseMaterial);
                command(player, "fill " + box(6, 1, 1, 6, 3, 5) + " " + houseMaterial);
                command(player, "fill " + box(0, 4, 0, 6, 4, 6) + " " + houseMaterial);
                beginVerification();
            }
            case ADD_DETAILS -> {
                if (!action.equals("add_door_and_windows")) return;
                placeDoor();
                command(player, "fill " + box(0, 2, 2, 0, 2, 4) + " minecraft:glass");
                command(player, "fill " + box(6, 2, 2, 6, 2, 4) + " minecraft:glass");
                command(player, "fill " + box(2, 2, 6, 4, 2, 6) + " minecraft:glass");
                beginVerification();
            }
            case PLACE_FURNACE -> {
                if (!action.equals("place_furnace")) return;
                furnacePosition = taskOrigin.offset(1, 1, 1);
                command(player, "setblock " + absolute(furnacePosition) +
                    " minecraft:furnace[facing=south]");
                beginVerification();
            }
            case LOAD_CHICKEN -> {
                if (action.equals("cook_one_chicken")) chickenQuantity = 1;
                else if (action.equals("cook_four_chickens")) chickenQuantity = 4;
                else return;
                setFurnaceItem(0, new ItemStack(Items.CHICKEN, chickenQuantity));
                beginVerification();
            }
            case LOAD_FUEL -> {
                if (action.equals("fuel_with_coal"))
                    setFurnaceItem(1, new ItemStack(Items.COAL));
                else if (action.equals("fuel_with_charcoal"))
                    setFurnaceItem(1, new ItemStack(Items.CHARCOAL));
                else return;
                beginVerification();
            }
            case COLLECT_CHICKEN -> {
                if (!action.equals("collect_now") && !action.equals("collect_and_finish")) return;
                transferCookedChicken();
                beginVerification();
            }
            default -> { }
        }
    }

    private String box(int x1, int y1, int z1, int x2, int y2, int z2) {
        return position(x1, y1, z1) + " " + position(x2, y2, z2);
    }

    private String position(int x, int y, int z) {
        return absolute(taskOrigin.offset(x, y, z));
    }

    private static String absolute(BlockPos position) {
        return position.getX() + " " + position.getY() + " " + position.getZ();
    }

    private static void command(LocalPlayer player, String command) {
        LOGGER.info("Decider command: /{}", command);
        player.connection.sendCommand(command);
    }

    private boolean bindOrValidateTaskSession(Minecraft client, LocalPlayer player) {
        IntegratedServer server = client.getSingleplayerServer();
        if (taskServer == null) {
            taskServer = server;
            taskDimension = client.level.dimension();
            taskPlayerId = player.getUUID();
            return true;
        }
        if (taskServer == server && taskDimension.equals(client.level.dimension()) &&
            taskPlayerId.equals(player.getUUID())) return true;
        failAndDisarm(player, "World or player changed during the task");
        return false;
    }

    private void beginVerification() {
        awaitingVerification = true;
        verificationTicks = 0;
    }

    private void verifyTaskPostcondition(Minecraft client, LocalPlayer player) {
        verificationTicks++;
        boolean verified = switch (taskPhase) {
            case PREPARE_SITE -> isHouseMaterial(client, taskOrigin) &&
                client.level.getBlockState(taskOrigin.above()).isAir() &&
                isHouseMaterial(client, taskOrigin.offset(6, 0, 6));
            case BUILD_SHELL -> isHouseMaterial(client, taskOrigin.offset(0, 2, 3)) &&
                isHouseMaterial(client, taskOrigin.offset(3, 4, 3)) &&
                client.level.getBlockState(taskOrigin.offset(3, 2, 3)).isAir();
            case ADD_DETAILS ->
                client.level.getBlockState(taskOrigin.offset(3, 1, 0)).is(Blocks.OAK_DOOR) &&
                client.level.getBlockState(taskOrigin.offset(3, 2, 0)).is(Blocks.OAK_DOOR) &&
                client.level.getBlockState(taskOrigin.offset(0, 2, 3)).is(Blocks.GLASS);
            case PLACE_FURNACE -> client.level.getBlockState(furnacePosition).is(Blocks.FURNACE);
            case LOAD_CHICKEN -> furnaceHas(Items.CHICKEN, 0);
            case LOAD_FUEL -> furnaceIsFueled();
            case COLLECT_CHICKEN -> Boolean.TRUE.equals(transferResult.get());
            default -> false;
        };
        if (taskPhase == TaskPhase.LOAD_CHICKEN || taskPhase == TaskPhase.LOAD_FUEL)
            requestFurnaceSnapshot();
        if (verified) {
            TaskPhase completed = taskPhase;
            awaitingVerification = false;
            verificationTicks = 0;
            taskPhase = switch (completed) {
                case PREPARE_SITE -> TaskPhase.BUILD_SHELL;
                case BUILD_SHELL -> TaskPhase.ADD_DETAILS;
                case ADD_DETAILS -> TaskPhase.PLACE_FURNACE;
                case PLACE_FURNACE -> TaskPhase.LOAD_CHICKEN;
                case LOAD_CHICKEN -> TaskPhase.LOAD_FUEL;
                case LOAD_FUEL -> TaskPhase.WAIT_FOR_COOKING;
                case COLLECT_CHICKEN -> TaskPhase.COMPLETE;
                default -> completed;
            };
            LOGGER.info("Verified task phase: {}", completed);
            if (taskPhase == TaskPhase.COMPLETE) completeTask(player);
        } else if (verificationTicks >= 100) {
            awaitingVerification = false;
            String message = "Postcondition timed out for " + taskPhase;
            player.sendSystemMessage(Component.literal("Decider error: " + message));
            LOGGER.error(message);
        }
    }

    private boolean isHouseMaterial(Minecraft client, BlockPos position) {
        Block expected = houseMaterial.equals("minecraft:cobblestone")
            ? Blocks.COBBLESTONE : Blocks.OAK_PLANKS;
        return client.level.getBlockState(position).is(expected);
    }

    private boolean furnaceHas(net.minecraft.world.item.Item item, int slot) {
        FurnaceSnapshot snapshot = furnaceSnapshot.get();
        if (snapshot == null) return false;
        return slot == 0 ? snapshot.rawChicken() > 0 : snapshot.coal() > 0;
    }

    private boolean furnaceIsFueled() {
        FurnaceSnapshot snapshot = furnaceSnapshot.get();
        return snapshot != null && (snapshot.coal() > 0 || snapshot.lit());
    }

    private void requestFurnaceSnapshot() {
        if (furnacePosition == null || taskServer == null ||
            !furnacePollPending.compareAndSet(false, true)) return;
        BlockPos position = furnacePosition.immutable();
        ResourceKey<Level> dimension = taskDimension;
        taskServer.execute(() -> {
            try {
                ServerLevel level = taskServer.getLevel(dimension);
                if (level != null && level.getBlockEntity(position) instanceof
                    AbstractFurnaceBlockEntity furnace) {
                    furnaceSnapshot.set(new FurnaceSnapshot(
                        count(furnace.getItem(0), Items.CHICKEN),
                        count(furnace.getItem(1), Items.COAL) +
                            count(furnace.getItem(1), Items.CHARCOAL),
                        count(furnace.getItem(2), Items.COOKED_CHICKEN),
                        level.getBlockState(position).getValue(BlockStateProperties.LIT)));
                }
            } finally {
                furnacePollPending.set(false);
            }
        });
    }

    private void setFurnaceItem(int slot, ItemStack stack) {
        BlockPos position = furnacePosition.immutable();
        ResourceKey<Level> dimension = taskDimension;
        taskServer.execute(() -> {
            ServerLevel level = taskServer.getLevel(dimension);
            if (level != null && level.getBlockEntity(position) instanceof
                AbstractFurnaceBlockEntity furnace) {
                furnace.setItem(slot, stack);
                furnace.setChanged();
                furnaceSnapshot.set(null);
            }
        });
    }

    private void placeDoor() {
        BlockPos lowerPosition = taskOrigin.offset(3, 1, 0).immutable();
        ResourceKey<Level> dimension = taskDimension;
        taskServer.execute(() -> {
            ServerLevel level = taskServer.getLevel(dimension);
            if (level == null) return;
            BlockState lower = Blocks.OAK_DOOR.defaultBlockState()
                .setValue(BlockStateProperties.HORIZONTAL_FACING, Direction.NORTH)
                .setValue(BlockStateProperties.DOUBLE_BLOCK_HALF, DoubleBlockHalf.LOWER);
            BlockState upper = lower.setValue(
                BlockStateProperties.DOUBLE_BLOCK_HALF, DoubleBlockHalf.UPPER);
            level.setBlock(lowerPosition, lower, Block.UPDATE_CLIENTS);
            level.setBlock(lowerPosition.above(), upper, Block.UPDATE_CLIENTS);
            level.updateNeighborsAt(lowerPosition, Blocks.OAK_DOOR);
            level.updateNeighborsAt(lowerPosition.above(), Blocks.OAK_DOOR);
        });
    }

    private void transferCookedChicken() {
        transferResult.set(null);
        BlockPos position = furnacePosition.immutable();
        ResourceKey<Level> dimension = taskDimension;
        UUID playerId = taskPlayerId;
        taskServer.execute(() -> {
            ServerLevel level = taskServer.getLevel(dimension);
            ServerPlayer player = taskServer.getPlayerList().getPlayer(playerId);
            if (level == null || player == null || !(level.getBlockEntity(position) instanceof
                AbstractFurnaceBlockEntity furnace)) {
                transferResult.set(false);
                return;
            }
            ItemStack cooked = furnace.removeItem(2, 1);
            boolean added = !cooked.isEmpty() && player.getInventory().add(cooked);
            if (!added && !cooked.isEmpty()) furnace.setItem(2, cooked);
            furnace.setChanged();
            transferResult.set(added);
        });
    }

    private static int count(ItemStack stack, net.minecraft.world.item.Item item) {
        return stack.getItem() == item ? stack.getCount() : 0;
    }

    private void completeTask(LocalPlayer player) {
        player.sendSystemMessage(Component.literal(
            "Decider task complete: house built and chicken cooked"));
        LOGGER.info("Decider house-and-chicken task complete");
        try {
            Files.deleteIfExists(FabricLoader.getInstance().getConfigDir()
                .resolve("decider-house-chicken.enabled"));
            Files.deleteIfExists(FabricLoader.getInstance().getConfigDir()
                .resolve("decider-autopilot.enabled"));
        } catch (IOException exception) {
            LOGGER.warn("Could not disarm completed task", exception);
        }
    }

    private void failAndDisarm(LocalPlayer player, String message) {
        player.sendSystemMessage(Component.literal("Decider error: " + message));
        LOGGER.error(message);
        try {
            Files.deleteIfExists(FabricLoader.getInstance().getConfigDir()
                .resolve("decider-house-chicken.enabled"));
            Files.deleteIfExists(FabricLoader.getInstance().getConfigDir()
                .resolve("decider-autopilot.enabled"));
        } catch (IOException exception) {
            LOGGER.warn("Could not disarm failed task", exception);
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

    private enum TaskPhase {
        CHOOSE_MATERIAL,
        PREPARE_SITE,
        BUILD_SHELL,
        ADD_DETAILS,
        PLACE_FURNACE,
        LOAD_CHICKEN,
        LOAD_FUEL,
        WAIT_FOR_COOKING,
        COLLECT_CHICKEN,
        COMPLETE
    }

    private record Decision(String action, double probability, Duration latency,
                            long generation) { }

    private record FurnaceSnapshot(int rawChicken, int coal, int cookedChicken,
                                   boolean lit) { }

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
