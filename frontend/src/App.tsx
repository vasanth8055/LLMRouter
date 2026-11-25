import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/layout/Layout";
import { Playground } from "./pages/Playground";
import { Analytics } from "./pages/Analytics";
import { Requests } from "./pages/Requests";
import { Settings } from "./pages/Settings";

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Navigate to="/playground" replace />} />
        <Route path="overview" element={<Analytics mode="overview" />} />
        <Route path="analytics" element={<Analytics mode="analytics" />} />
        <Route path="playground" element={<Playground />} />
        <Route path="requests" element={<Requests />} />
        <Route path="settings" element={<Settings />} />
      </Route>
      <Route path="*" element={<Navigate to="/playground" replace />} />
    </Routes>
  );
}
