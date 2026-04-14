import unittest

from ghostext.packet import (
    AEAD_TAG_LEN,
    INTERNAL_HEADER_SIZE,
    InternalHeader,
    packet_bootstrap_size,
    split_packet,
)


class PacketTests(unittest.TestCase):
    def test_internal_header_pack_roundtrip(self) -> None:
        header = InternalHeader.build(
            body_ciphertext_len=23,
            config_fingerprint=0x1234,
        )
        packed = header.pack()
        self.assertEqual(len(packed), INTERNAL_HEADER_SIZE)
        unpacked = InternalHeader.unpack(packed)
        self.assertEqual(unpacked, header)

    def test_split_packet_validates_bootstrap_lengths(self) -> None:
        salt_len = 16
        nonce_len = 12
        bootstrap_len = packet_bootstrap_size(salt_len, nonce_len)
        self.assertEqual(
            bootstrap_len,
            salt_len + nonce_len + INTERNAL_HEADER_SIZE + AEAD_TAG_LEN,
        )

        packet = (b"x" * bootstrap_len) + b"body"
        bootstrap, body = split_packet(packet, salt_len=salt_len, nonce_len=nonce_len)
        self.assertEqual(len(bootstrap), bootstrap_len)
        self.assertEqual(body, b"body")


if __name__ == "__main__":
    unittest.main()

