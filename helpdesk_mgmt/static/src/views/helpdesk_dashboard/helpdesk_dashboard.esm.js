import {Component, onWillStart, usePlugin} from "@odoo/owl";
import {ActionPlugin} from "@web/webclient/actions/action_plugin";
import {SIZES} from "@web/core/ui/ui_utils";
import {UIPlugin} from "@web/core/ui/ui_plugin";
import {ViewButton} from "@web/views/view_button/view_button";
import {useService} from "@web/core/utils/hooks";

export class HelpdeskDashboard extends Component {
    static template = "helpdesk_mgmt.HelpdeskDashboard";
    static components = {ViewButton};
    setup() {
        this.orm = useService("orm");
        this.action = usePlugin(ActionPlugin);
        this.ui = usePlugin(UIPlugin);
        onWillStart(async () => {
            this.helpdeskData = await this.orm.call(
                "helpdesk.ticket.team",
                "retrieve_dashboard"
            );
        });
    }
    clickParams(section) {
        if (section.action) {
            return {name: section.action, type: "action"};
        }
        return {};
    }

    get gridTemplateColumns() {
        // Reading the `size` signal re-renders the component on resize
        switch (this.ui.size()) {
            case SIZES.XS:
                return 2;
            case SIZES.SM:
                return 3;
            case SIZES.XXL:
                return 6;
            default:
                return 4;
        }
    }
}
