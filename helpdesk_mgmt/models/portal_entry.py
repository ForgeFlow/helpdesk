# License AGPL-3.0 or later (http://www.gnu.org/licenses/agpl).
from odoo import models


class PortalEntry(models.Model):
    _inherit = "portal.entry"

    def _filter_visible_portal_cards(self):
        # Always show the tickets card, so that customers can open their
        # first ticket from the portal home
        tickets_entry = self.env.ref(
            "helpdesk_mgmt.portal_entry_tickets", raise_if_not_found=False
        )
        return super()._filter_visible_portal_cards() | (self & tickets_entry)
