package com.kf7mxe.brisingr.data

import com.lightningkite.lightningserver.auth.require
import com.lightningkite.lightningserver.definition.builder.ServerBuilder
import com.lightningkite.lightningserver.runtime.ServerRuntime
import com.lightningkite.lightningserver.typed.AuthAccess
import com.lightningkite.lightningserver.typed.ModelRestEndpoints
import com.lightningkite.lightningserver.typed.modelInfo
import com.kf7mxe.brisingr.AppRelease
import com.kf7mxe.brisingr.Server
import com.kf7mxe.brisingr.User
import com.kf7mxe.brisingr.UserAuth
import com.kf7mxe.brisingr.UserAuth.RoleCache.userRole
import com.kf7mxe.brisingr.UserRole
import com.lightningkite.services.database.Condition
import com.lightningkite.services.database.ModelPermissions

object AppReleaseEndpoints : ServerBuilder() {

    val info = Server.database.modelInfo(
        auth = UserAuth.require(),
        permissions = { permissions(this) },
    )
    val rest = path include ModelRestEndpoints(info)

    context(server: ServerRuntime)
    suspend fun permissions(auth: AuthAccess<User>): ModelPermissions<AppRelease> {
        return if (auth.userRole() < UserRole.Admin) {
            ModelPermissions(read = Condition.Always)
        } else {
            ModelPermissions.allowAll()
        }
    }
}