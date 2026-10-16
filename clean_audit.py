"""
One-off cleanup: removes audit rows written by the OLD prototype (status 'IMMUNIZED'),
which came from a hard-coded rule rather than a measurement. Real runs are labelled
DISRUPTED, PARTIAL or MINIMAL and are kept. Delete this file after running it once.
"""
import sqlite3

conn = sqlite3.connect("users.db")
cur = conn.execute("DELETE FROM audit_logs WHERE defense_status = 'IMMUNIZED'")
conn.commit()
print("removed", cur.rowcount, "old test rows")
conn.close()
