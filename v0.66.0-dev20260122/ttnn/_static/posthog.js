(function (t, e) {
  var o, n, p, r;
  if (!e.__SV) {
    window.posthog = e;
    e._i = [];
    e.init = function (i, s, a) {
      function g(t, e) {
        var o = e.split(".");
        if (o.length === 2) {
          t = t[o[0]];
          e = o[1];
        }
        t[e] = function () {
          t.push([e].concat(Array.prototype.slice.call(arguments, 0)));
        };
      }
      p = t.createElement("script");
      p.type = "text/javascript";
      p.crossOrigin = "anonymous";
      p.async = true;
      p.src =
        s.api_host.replace(".i.posthog.com", "-assets.i.posthog.com") +
        "/static/array.js";
      r = t.getElementsByTagName("script")[0];
      r.parentNode.insertBefore(p, r);

      var u = e;
      if (a !== undefined) {
        u = e[a] = [];
      } else {
        a = "posthog";
      }

      u.people = u.people || [];

      u.toString = function (t) {
        var e = "posthog";
        if (a !== "posthog") {
          e += "." + a;
        }
        if (!t) {
          e += " (stub)";
        }
        return e;
      };

      u.people.toString = function () {
        return u.toString(1) + ".people (stub)";
      };

      var methods = (
        "init Ie Ts Ms Ee Es Rs capture Ge calculateEventProperties Os register " +
        "register_once register_for_session unregister unregister_for_session js getFeatureFlag " +
        "getFeatureFlagPayload isFeatureEnabled reloadFeatureFlags updateEarlyAccessFeatureEnrollment " +
        "getEarlyAccessFeatures on onFeatureFlags onSurveysLoaded onSessionId getSurveys " +
        "getActiveMatchingSurveys renderSurvey canRenderSurvey canRenderSurveyAsync identify " +
        "setPersonProperties group resetGroups setPersonPropertiesForFlags resetPersonPropertiesForFlags " +
        "setGroupPropertiesForFlags resetGroupPropertiesForFlags reset get_distinct_id getGroups " +
        "get_session_id get_session_replay_url alias set_config startSessionRecording " +
        "stopSessionRecording sessionRecordingStarted captureException loadToolbar get_property " +
        "getSessionProperty Ds Fs createPersonProfile Ls Ps opt_in_capturing opt_out_capturing " +
        "has_opted_in_capturing has_opted_out_capturing clear_opt_in_out_capturing Cs debug I As " +
        "getPageViewId captureTraceFeedback captureTraceMetric"
      ).split(" ");

      for (n = 0; n < methods.length; n++) {
        g(u, methods[n]);
      }

      e._i.push([i, s, a]);
    };
    e.__SV = 1;
  }
})(document, window.posthog || []);

API_KEY = "phc_9LMRmHrCFvQNvDkPDjYBP5dZ6WchZ5bcM6T4Qj6tb0U";

posthog.init(API_KEY, {
  api_host: "https://us.i.posthog.com",
  defaults: "2025-05-24",
  person_profiles: "identified_only", // or 'always'
});
