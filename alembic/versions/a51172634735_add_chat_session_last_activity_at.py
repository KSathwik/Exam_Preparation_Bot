"""add chat_session last_activity_at

Revision ID: a51172634735
Revises: 1968abf5b1d7
Create Date: 2026-08-04 15:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a51172634735'
down_revision: Union[str, Sequence[str], None] = '1968abf5b1d7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('chat_sessions', schema=None) as batch_op:
        batch_op.add_column(sa.Column('last_activity_at', sa.DateTime(), nullable=True))

    # Backfill existing rows so sort order doesn't reset for conversations
    # created before this column existed — updated_at is the closest
    # available proxy for "last real activity" at migration time, since the
    # bug this column fixes (rename bumping updated_at) only starts
    # affecting rows going forward from here.
    op.execute("UPDATE chat_sessions SET last_activity_at = updated_at WHERE last_activity_at IS NULL")


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('chat_sessions', schema=None) as batch_op:
        batch_op.drop_column('last_activity_at')
