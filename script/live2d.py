import asyncio
import pyvts


async def main():
    # Create an instance of the VTube Studio API client
    vts_client = pyvts.vts()

    # Connect to VTube Studio
    await vts_client.connect()

    # Here you can implement the functionality you want to achieve with the API
    # For example, you can add new tracking parameters or interact with Live2D avatars

    # Close the connection to VTube Studio
    await vts_client.close()


if __name__ == "__main__":
    asyncio.run(main())
