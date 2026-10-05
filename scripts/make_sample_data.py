"""Regenerates the small original sample documents in sample_data/ (TXT, DOCX, PDF)."""
from pathlib import Path

from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "sample_data"
OUT.mkdir(exist_ok=True)

NETWORKING = """Computer Networking Basics

The Transmission Control Protocol (TCP) is a connection-oriented transport protocol. Before any application data is exchanged, the two hosts must agree to talk to each other. TCP establishes communication using a three-way handshake: the client sends a SYN segment, the server answers with SYN-ACK, and the client confirms with an ACK.

TCP also provides reliable delivery. Every segment carries a sequence number, the receiver acknowledges what it has received, and missing segments are retransmitted. Flow control and congestion control keep a fast sender from overwhelming a slow receiver or the network.

The Domain Name System (DNS) works like the phone book of the internet. When you type a website name into a browser, a DNS resolver translates the human-readable name into the numeric IP address of the server, often by asking a chain of recursive, root, top-level-domain and authoritative name servers.

The User Datagram Protocol (UDP) is connectionless. It skips the handshake and retransmission, which makes it faster but unreliable. It suits live video, voice calls and online games where a late packet is worthless.
"""
(OUT / "networking_basics.txt").write_text(NETWORKING, encoding="utf-8")

d = Document()
d.add_heading("Algorithms and Data Structures Notes", 1)
for text in [
    "Merge sort is a divide-and-conquer algorithm. It splits the array into two halves, sorts each half recursively and then merges the two sorted halves into one sorted array. Its running time is O(n log n) in every case, at the cost of O(n) extra memory.",
    "Quick sort picks a pivot element and partitions the other elements into those smaller and larger than the pivot. On average it runs in O(n log n), but a poor pivot choice can degrade it to O(n^2).",
    "A hash table stores key-value pairs in an array. A hash function maps each key to an index, so a lookup does not have to scan every element; on average insertion, deletion and lookup take constant time. Collisions are handled with chaining or open addressing.",
    "A binary search tree keeps keys ordered so that searching, inserting and deleting take O(log n) time when the tree is balanced.",
]:
    d.add_paragraph(text)
t = d.add_table(rows=3, cols=2)
for r, (a, b) in enumerate([("Structure", "Typical lookup"), ("Hash table", "O(1) average"), ("Balanced BST", "O(log n)")]):
    t.cell(r, 0).text, t.cell(r, 1).text = a, b
d.save(OUT / "algorithms_notes.docx")

styles = getSampleStyleSheet()
pdf = SimpleDocTemplate(str(OUT / "biology_primer.pdf"), pagesize=A4, title="Biology Primer")
story = [
    Paragraph("Biology Primer", styles["Title"]),
    Paragraph("Photosynthesis is the process by which plants, algae and some bacteria convert light energy into chemical energy. Inside the chloroplasts, chlorophyll absorbs sunlight and uses it to turn carbon dioxide and water into glucose, releasing oxygen as a by-product.", styles["BodyText"]),
    Spacer(1, 12),
    Paragraph("The light-dependent reactions happen in the thylakoid membranes and produce ATP and NADPH. The Calvin cycle then uses that energy in the stroma to build sugar molecules.", styles["BodyText"]),
    PageBreak(),
    Paragraph("Cell Respiration", styles["Heading2"]),
    Paragraph("Cellular respiration releases the energy stored in glucose. In the mitochondria, glucose is broken down with oxygen to produce ATP, the molecule cells use as an energy currency, along with carbon dioxide and water.", styles["BodyText"]),
]
pdf.build(story)
print("Wrote", [p.name for p in OUT.iterdir()])
