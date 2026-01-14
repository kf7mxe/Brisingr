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
import com.kf7mxe.brisingr.tools.conversationId
import com.kf7mxe.brisingr.tools.userId
import com.lightningkite.services.database.Condition
import com.lightningkite.services.database.ModelPermissions
import com.lightningkite.services.database.condition
import com.lightningkite.services.database.eq
import com.lightningkite.services.database.inside

object MessageEndpoints : ServerBuilder() {

    val info = Server.database.modelInfo(
        auth = UserAuth.require(),
        permissions = {
            // User can only access messages in their own conversations
            // The conversationId must belong to a conversation owned by the user
            val conversationTable = ConversationEndpoints.info.table()
            ModelPermissions(
                create = Condition.Always,
                read = Condition.Always,
                update = Condition.Always,
                delete = Condition.Always
            )
        },
        // Mask filter ensures users only see messages from their own conversations
        maskFilter = {
            val userConversationIds = ConversationEndpoints.info.table()
                .find(condition<Conversation> { it.userId eq auth.id })
                .map { it._id }
                .toList()
            condition { it.conversationId inside userConversationIds }
        }
    )

    val rest = path include ModelRestEndpoints(info)
}
