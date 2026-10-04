// Copyright 2024 Nitrokey GmbH
// License AGPL-3.0 or later (https://www.gnu.org/licenses/agpl).

import {_t} from "@web/core/l10n/translation";
import {registry} from "@web/core/registry";

registry.category("builder.form_editor_actions").add("create_helpdesk_ticket", {
    fields: [
        {
            name: "category_id",
            type: "many2one",
            relation: "helpdesk.ticket.category",
            string: _t("Category"),
            title: _t("Assign tickets to a category."),
        },
        {
            name: "team_id",
            type: "many2one",
            relation: "helpdesk.ticket.team",
            string: _t("Support Team"),
            title: _t("Assign tickets to a support team."),
        },
    ],
});
