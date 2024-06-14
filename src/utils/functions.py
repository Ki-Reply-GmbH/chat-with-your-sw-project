"""
This module provides utility functions for encoding and decoding strings to and from base64.

The `encode_to_base64` function takes a string or a dictionary, converts it to a string if it's
not already, and encodes it to base64.

The `decode_from_base64` function takes a base64 encoded string and decodes it.
"""
import base64


def encode_to_base64(input_string):
    """
    Encodes a string or a dictionary to base64.

    This function checks if the input is a dictionary. If it is, it converts it to a string. It
    then encodes the input string to base64 and returns the encoded string.

    Args:
        input_string (str or dict): The string or dictionary to be encoded.

    Returns:
        str: The base64 encoded string.
    """
    # Check if input is a dictionary
    if isinstance(input_string, dict):
        # Convert dictionary to string
        input_string = str(input_string)

    # Encode the input string to base64
    encoded_bytes = base64.b64encode(input_string.encode("utf-8"))
    # Convert bytes to string and return
    encoded_string = encoded_bytes.decode("utf-8")
    return encoded_string


def decode_from_base64(encoded_string):
    """
    Decodes a base64 encoded string.

    This function takes a base64 encoded string, decodes it, and returns the decoded string.

    Args:
        encoded_string (str): The base64 encoded string to be decoded.

    Returns:
        str: The decoded string.
    """
    # Decode the base64 encoded string
    decoded_bytes = base64.b64decode(encoded_string.encode("utf-8"))
    # Convert bytes to string
    decoded_string = decoded_bytes.decode("utf-8")

    return decoded_string
