#!/usr/bin/env python3
"""Fetch plain-text Minecraft Wiki articles for local model training.

Reproducible data fetcher. Uses only the Python standard library.
Output: data/mc.txt (concatenated article plain text) and
.agent-work/tmp/mcwiki/*.txt (per-page cache).

Source: https://minecraft.wiki (CC BY-NC-SA 3.0). This corpus is used
only for local, non-commercial language-model training.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

API = "https://minecraft.wiki/api.php"
OUT = "data/mc.txt"
CACHE = ".agent-work/tmp/mcwiki"

# Category crawls add broad coverage beyond the hand-picked pages below.
# MC_MAX_PAGES caps the total so a local run stays reproducible and the
# repository does not grow without bound.
CATEGORIES = [
    ("Blocks", 260),
    ("Items", 260),
    ("Mobs", 80),
    ("Food", 80),
    ("Tools", 80),
    ("Redstone", 80),
    ("Enchantments", 60),
    ("Biomes", 30),
    ("Structures", 40),
]

PAGES = [
    # Core game
    "Minecraft", "Java Edition", "Bedrock Edition", "Gameplay", "Controls",
    "Difficulty", "Game mode", "Survival", "Creative", "Adventure", "Spectator",
    "Health", "Hunger", "Experience", "Advancement", "Statistics",
    # World
    "Overworld", "The Nether", "The End", "Biome", "Seed (level generation)",
    "World generation", "Chunk", "Structure", "Village", "Stronghold",
    "Nether Fortress", "End City", "Woodland Mansion", "Ocean Monument",
    "Ancient City", "Trial Chambers", "Mineshaft", "Ruined Portal",
    "Desert Temple", "Jungle Temple", "Witch Hut", "Pillager Outpost",
    # Blocks
    "Block", "Dirt", "Grass Block", "Stone", "Cobblestone", "Oak Log",
    "Oak Planks", "Crafting Table", "Furnace", "Chest", "Barrel", "Bed",
    "Torch", "Ladder", "Door", "Glass", "Sand", "Gravel", "Water", "Lava",
    "Obsidian", "Bedrock", "Netherrack", "Soul Sand", "Glowstone", "End Stone",
    "Purpur Block", "Wool", "Concrete", "Terracotta", "Bookshelf",
    "Enchanting Table", "Anvil", "Brewing Stand", "Cauldron", "Beacon",
    "Spawner", "Command Block", "Structure Block", "Barrier", "Light",
    # Ores and items
    "Ore", "Coal Ore", "Iron Ore", "Copper Ore", "Gold Ore", "Redstone Ore",
    "Lapis Lazuli Ore", "Diamond Ore", "Emerald Ore", "Nether Quartz Ore",
    "Ancient Debris", "Coal", "Iron Ingot", "Copper Ingot", "Gold Ingot",
    "Redstone Dust", "Lapis Lazuli", "Diamond", "Emerald", "Netherite",
    "Netherite Ingot", "Netherite Scrap", "Stick", "String", "Feather",
    "Gunpowder", "Flint", "Leather", "Paper", "Book", "Sugar", "Bone",
    "Slimeball", "Ender Pearl", "Eye of Ender", "Blaze Rod", "Blaze Powder",
    # Tools, combat, armor
    "Tool", "Crafting", "Sword", "Pickaxe", "Axe", "Shovel", "Hoe",
    "Bow", "Crossbow", "Arrow", "Trident", "Shield", "Armor", "Helmet",
    "Chestplate", "Leggings", "Boots", "Durability", "Enchanting",
    "Enchantment", "Anvil mechanics", "Potion", "Brewing", "Status effect",
    # Mobs
    "Mob", "Creeper", "Zombie", "Skeleton", "Spider", "Cave Spider",
    "Enderman", "Slime", "Magma Cube", "Ghast", "Blaze", "Piglin",
    "Zombified Piglin", "Hoglin", "Strider", "Wither Skeleton",
    "Ender Dragon", "Wither", "Elder Guardian", "Guardian", "Shulker",
    "Villager", "Wandering Trader", "Iron Golem", "Snow Golem", "Wolf",
    "Cat", "Horse", "Donkey", "Mule", "Llama", "Bee", "Fox", "Panda",
    "Axolotl", "Goat", "Frog", "Allay", "Pillager", "Vindicator",
    "Evoker", "Ravager", "Witch", "Illager", "Raid", "Villager trading",
    # Farming and food
    "Farming", "Wheat", "Carrot", "Potato", "Beetroot", "Sugar Cane",
    "Pumpkin", "Melon", "Cactus", "Bamboo", "Kelp", "Cocoa Beans",
    "Food", "Bread", "Apple", "Golden Apple", "Cooked Beef", "Steak",
    "Cooked Porkchop", "Cooked Chicken", "Cooked Mutton", "Cooked Rabbit",
    "Fish", "Mushroom Stew", "Cake", "Cookie", "Honey Bottle",
    # Redstone and technical
    "Redstone", "Redstone circuit", "Redstone Torch", "Redstone Repeater",
    "Redstone Comparator", "Piston", "Sticky Piston", "Observer",
    "Hopper", "Dropper", "Dispenser", "Minecart", "Rail", "Powered Rail",
    "Detector Rail", "Activator Rail", "TNT", "Tripwire", "Pressure Plate",
    "Button", "Lever", "Daylight Detector", "Target Block", "Note Block",
    # Commands and servers
    "Command", "Commands/give", "Commands/tp", "Commands/gamemode",
    "Commands/time", "Commands/weather", "Commands/summon", "Commands/enchant",
    "Commands/effect", "Commands/setblock", "Commands/fill", "Commands/clone",
    "Commands/execute", "Command selector", "Data pack", "Resource pack",
    "Server", "Multiplayer", "Server.properties", "LAN",
    # Misc systems
    "Trading", "Villager trading", "Scoreboard", "Team", "Bossbar",
    "Damage", "Drowning", "Fall damage", "Fire", "Explosion",
    "Spawn", "Respawn", "Nether portal", "End portal", "End Gateway",
    "Elytra", "Firework Rocket", "Boat", "Minecart", "Map", "Compass",
    "Clock", "Bed", "Respawn Anchor", "Lodestone", "Conduit", "Beacon",
]


def fetch(title):
    # Whole-article extracts are limited to exlimit=1 per request.
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "format": "json",
        "formatversion": "2",
        "exlimit": "1",
        "titles": title,
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "UltronCorpusFetcher/1.0 (local model training)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def cache_name(title):
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title).strip()
    return os.path.join(CACHE, safe + ".txt")


def category_members(category, limit):
    """Return up to `limit` article titles from a category, newest crawl only."""
    out = []
    cont = None
    while len(out) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:" + category,
            "cmlimit": "500",
            "format": "json",
            "formatversion": "2",
        }
        if cont:
            params["cmcontinue"] = cont
        url = API + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "UltronCorpusFetcher/1.0 (local model training)"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print("category %s failed: %s" % (category, e), file=sys.stderr)
            break
        for m in data.get("query", {}).get("categorymembers", []):
            title = m.get("title", "")
            ns = m.get("ns", 0)
            if ns != 0:
                continue
            if not title or "/" in title or "(" in title or ")" in title:
                continue
            if title.lower().startswith(("tutorial:", "minecraft wiki:", "category:", "template:")):
                continue
            out.append(title)
            if len(out) >= limit:
                break
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        time.sleep(0.2)
    return out


def ordered_titles():
    seen = set()
    titles = []
    for title in PAGES:
        if title not in seen:
            seen.add(title)
            titles.append(title)
    for category, limit in CATEGORIES:
        for title in category_members(category, limit):
            if title not in seen:
                seen.add(title)
                titles.append(title)
    cap = int(os.environ.get("MC_MAX_PAGES", "1000"))
    return titles[:cap]


def main():
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pages = {}
    missing = []
    titles = ordered_titles()
    for idx, requested in enumerate(titles):
        cache_path = cache_name(requested)
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            with open(cache_path, encoding="utf-8") as f:
                pages[requested] = f.read()
            continue
        try:
            data = fetch(requested)
        except Exception as e:
            print("fetch failed for %s: %s" % (requested, e), file=sys.stderr)
            time.sleep(1.0)
            missing.append(requested)
            continue
        found = None
        for p in data.get("query", {}).get("pages", []):
            text = p.get("extract", "")
            if text:
                found = (p.get("title", requested), text)
                break
        if not found:
            missing.append(requested)
            continue
        actual, text = found
        # Keep articles bounded; the tail of a huge page rarely adds new facts.
        text = text[:24000]
        pages[requested] = text
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("[%d/%d] %s -> %s (%d chars)" % (idx + 1, len(titles), requested, actual, len(text)), file=sys.stderr)
        time.sleep(0.25)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("*** MINECRAFT JAVA EDITION KNOWLEDGE CORPUS ***\n")
        f.write("Plain-text extracts from minecraft.wiki, used for local model training.\n")
        f.write("Source: https://minecraft.wiki (CC BY-NC-SA 3.0).\n\n")
        for title in titles:
            text = pages.get(title)
            if not text:
                continue
            f.write("\n\n===== %s =====\n\n" % title)
            f.write(text.strip())
            f.write("\n")
    print("wrote %s: %d pages, %d bytes" % (OUT, len(pages), os.path.getsize(OUT)), file=sys.stderr)
    if missing:
        print("missing: %s" % ", ".join(missing), file=sys.stderr)


if __name__ == "__main__":
    main()
