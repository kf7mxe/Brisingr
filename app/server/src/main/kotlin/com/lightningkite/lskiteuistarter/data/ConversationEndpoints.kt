package com.kf7mxe.brisingr.data

import com.lightningkite.lightningserver.auth.id
import com.lightningkite.lightningserver.auth.require
import com.lightningkite.lightningserver.definition.builder.ServerBuilder
import com.lightningkite.lightningserver.typed.ModelRestEndpoints
import com.lightningkite.lightningserver.typed.auth
import com.lightningkite.lightningserver.typed.modelInfo
import com.kf7mxe.brisingr.Server
import com.kf7mxe.brisingr.UserAuth
import com.kf7mxe.brisingr.tools.Conversation
import com.kf7mxe.brisingr.tools.userId
import com.lightningkite.services.database.ModelPermissions
import com.lightningkite.services.database.condition
import com.lightningkite.services.database.eq

object ConversationEndpoints : ServerBuilder() {

    val info = Server.database.modelInfo(
        auth = UserAuth.require(),
        permissions = {
            val ownConversations = condition<Conversation> { it.userId eq auth.id }
            ModelPermissions(
                create = ownConversations,
                read = ownConversations,
                update = ownConversations,
                delete = ownConversations
            )
        }
    )

    val rest = path include ModelRestEndpoints(info)
}
