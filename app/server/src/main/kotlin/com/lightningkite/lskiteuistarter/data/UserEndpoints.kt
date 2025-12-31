package com.kf7mxe.brisingr.data

import com.lightningkite.lightningserver.auth.id
import com.lightningkite.lightningserver.auth.require
import com.lightningkite.lightningserver.definition.builder.ServerBuilder
import com.lightningkite.lightningserver.typed.ModelRestEndpoints
import com.lightningkite.lightningserver.typed.auth
import com.lightningkite.lightningserver.typed.modelInfo
import com.lightningkite.lightningserver.typed.startupOnce
import com.kf7mxe.brisingr.Server
import com.kf7mxe.brisingr.User
import com.kf7mxe.brisingr.UserAuth
import com.kf7mxe.brisingr.UserAuth.RoleCache.userRole
import com.kf7mxe.brisingr.UserRole
import com.kf7mxe.brisingr._id
import com.kf7mxe.brisingr.email
import com.kf7mxe.brisingr.role
import com.lightningkite.services.database.Condition
import com.lightningkite.services.database.ModelPermissions
import com.lightningkite.services.database.condition
import com.lightningkite.services.database.eq
import com.lightningkite.services.database.insertOne
import com.lightningkite.services.database.inside
import com.lightningkite.services.database.or
import com.lightningkite.services.database.updateRestrictions
import com.lightningkite.toEmailAddress
import kotlin.uuid.Uuid

object UserEndpoints : ServerBuilder() {

    val info = Server.database.modelInfo(
        auth = UserAuth.require(),
        permissions = {
            val allowedRoles = UserRole.entries.filter { it <= auth.userRole() }
            val admin: Condition<User> =
                if (this.auth.userRole() >= UserRole.Admin) condition { it.role inside allowedRoles } else Condition.Never
            val self = condition<User> { it._id eq auth.id }
            ModelPermissions(
                create = admin,
                read = admin or self,
                update = admin or self,
                updateRestrictions = updateRestrictions {
                    it.role.requires(admin) { it.inside(allowedRoles) }
                },
                delete = admin or self,
            )
        }
    )

    val rest = path include ModelRestEndpoints(info)
//    val socketUpdates = ModelRestUpdatesWebsocket(path, Server.database, info)

    val initAdminUser = path.path("initAdminUser") bind startupOnce(Server.database) {
        println("Adding user")
        val email = "joseph+root@lightningkite.com".toEmailAddress()
        info.table().deleteMany(condition { it.email.eq(email) })
        info.table().insertOne(
            User(
                _id = Uuid.Companion.fromLongs(0L, 10L),
                email = email,
                name = "Joseph Root",
                role = UserRole.Root
            )
        )
    }
}