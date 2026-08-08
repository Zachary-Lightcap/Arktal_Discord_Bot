# Imports
import cv2
import discord
import numpy as np
from discord.ext import commands
import logging
from dotenv import load_dotenv
from arktal_beads import analyze_image
import os
import webserver
import io

# Load environment and get discord token
load_dotenv()
token = os.getenv("DISCORD_TOKEN")

# Initialize bot variables
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    synced = await bot.tree.sync()
    print(f"Synced: {len(synced)} slash commands")
    print(f"Ready to go, {bot.user.name}!")

@bot.event
async def on_member_join(member):
    channel = member.guild.system_channel

    if channel is not None:
        await channel.send(f"Welcome to the server {member.name}, excited to help!")

@bot.command()
async def hello(ctx):
    await ctx.send(f"Hello {ctx.author.mention}!")

@bot.tree.command(name="bead", description="Create a image with bead's picked out")
async def bead(
        interaction: discord.Interaction,
        image: discord.Attachment
):
    # Defer the response
    await interaction.response.defer()

    # Read the uploaded image to bytes
    image_bytes = await image.read()

    # Convert to usable numpy array
    np_image = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode array with OpenCV
    img = cv2.imdecode(np_image, cv2.IMREAD_UNCHANGED)

    # Get the predicted image
    predicted_img = analyze_image(img)

    # Encode the OpenCV image as a PNG
    success, buffer = cv2.imencode('.png', predicted_img)

    if not success:
        await interaction.followup.send("Failed to create the output image.", ephemeral=True)
        return

    file = discord.File(
        io.BytesIO(buffer.tobytes()),
        filename="predicted_image.png"
    )

    # Send back the image
    await interaction.followup.send("Here you go!",
                    file=file)


@bot.command()
async def poll(ctx, *, question):
    embed = discord.Embed(title="New Poll", description=question)
    poll_message = await ctx.send(embed=embed)
    await poll_message.add_reaction("👍")
    await poll_message.add_reaction("👎")



@bot.command()
async def bug(ctx):
    await ctx.send("I'm trying my best 🥺")

webserver.keep_alive()
bot.run(token, log_handler=handler, log_level=logging.DEBUG)
