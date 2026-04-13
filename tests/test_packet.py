import unittest

from hidetext.packet import HEADER_SIZE, PacketHeader, split_packet


class PacketTests(unittest.TestCase):
    def test_header_pack_roundtrip(self) -> None:
        header = PacketHeader.build(
            salt_len=16,
            nonce_len=12,
            ciphertext_len=23,
            config_fingerprint=0x1234,
        )
        packed = header.pack()
        self.assertEqual(len(packed), HEADER_SIZE)
        unpacked = PacketHeader.unpack(packed)
        self.assertEqual(unpacked, header)

    def test_split_packet_validates_lengths(self) -> None:
        header = PacketHeader.build(
            salt_len=16,
            nonce_len=12,
            ciphertext_len=32,
            config_fingerprint=99,
        )
        packet = header.pack() + (b"x" * header.body_len)
        unpacked_header, body = split_packet(packet)
        self.assertEqual(unpacked_header, header)
        self.assertEqual(len(body), header.body_len)


if __name__ == "__main__":
    unittest.main()

