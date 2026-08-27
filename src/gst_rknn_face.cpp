#include "face_tracker.h"
#include "gst_rknn_face_meta.h"

#include <gst/video/gstvideofilter.h>

#include <cstring>
#include <exception>

#ifndef PACKAGE
#define PACKAGE "rknn-face-points"
#endif

extern "C" {

static gboolean rknn_face_meta_init(GstMeta* meta, gpointer, GstBuffer*) {
  auto* face = reinterpret_cast<GstRknnFaceMeta*>(meta);
  face->found = FALSE;
  face->landmark_count = 0;
  face->score = 0;
  face->tongue_out = 0;
  face->npu_ms = 0;
  std::memset(face->bbox, 0, sizeof(face->bbox));
  std::memset(face->head_pose_deg, 0, sizeof(face->head_pose_deg));
  std::memset(face->landmarks, 0, sizeof(face->landmarks));
  std::memset(face->blendshapes, 0, sizeof(face->blendshapes));
  return TRUE;
}

static gboolean rknn_face_meta_transform(GstBuffer* dest, GstMeta* meta, GstBuffer*, GQuark, gpointer) {
  auto* src = reinterpret_cast<GstRknnFaceMeta*>(meta);
  auto* out = gst_buffer_add_rknn_face_meta(dest);
  out->found = src->found;
  out->landmark_count = src->landmark_count;
  out->score = src->score;
  out->tongue_out = src->tongue_out;
  out->npu_ms = src->npu_ms;
  std::memcpy(out->bbox, src->bbox, sizeof(src->bbox));
  std::memcpy(out->head_pose_deg, src->head_pose_deg, sizeof(src->head_pose_deg));
  std::memcpy(out->landmarks, src->landmarks, sizeof(src->landmarks));
  std::memcpy(out->blendshapes, src->blendshapes, sizeof(src->blendshapes));
  return TRUE;
}

GType gst_rknn_face_meta_api_get_type(void) {
  static gsize type = 0;
  static const gchar* tags[] = {GST_META_TAG_VIDEO_STR, nullptr};
  if (g_once_init_enter(&type)) {
    GType value = gst_meta_api_type_register("GstRknnFaceMetaAPI", tags);
    g_once_init_leave(&type, value);
  }
  return static_cast<GType>(type);
}

const GstMetaInfo* gst_rknn_face_meta_get_info(void) {
  static gsize info = 0;
  if (g_once_init_enter(&info)) {
    const GstMetaInfo* value = gst_meta_register(
        GST_RKNN_FACE_META_API_TYPE, "GstRknnFaceMeta", sizeof(GstRknnFaceMeta),
        rknn_face_meta_init, nullptr, rknn_face_meta_transform);
    g_once_init_leave(&info, reinterpret_cast<gsize>(value));
  }
  return reinterpret_cast<const GstMetaInfo*>(info);
}

GstRknnFaceMeta* gst_buffer_add_rknn_face_meta(GstBuffer* buffer) {
  return reinterpret_cast<GstRknnFaceMeta*>(gst_buffer_add_meta(buffer, GST_RKNN_FACE_META_INFO, nullptr));
}

}  // extern "C"

typedef struct _GstRknnFaceMesh {
  GstVideoFilter parent;
  gchar* model_dir;
  gboolean draw;
  rknn_face::Tracker* tracker;
} GstRknnFaceMesh;

typedef struct _GstRknnFaceMeshClass {
  GstVideoFilterClass parent_class;
} GstRknnFaceMeshClass;

#define GST_TYPE_RKNN_FACE_MESH (gst_rknn_face_mesh_get_type())
#define GST_RKNN_FACE_MESH(obj) (reinterpret_cast<GstRknnFaceMesh*>(obj))

G_DEFINE_TYPE(GstRknnFaceMesh, gst_rknn_face_mesh, GST_TYPE_VIDEO_FILTER)

enum {
  PROP_0,
  PROP_MODEL_DIR,
  PROP_DRAW,
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw, format=(string)RGB"));
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw, format=(string)RGB"));

static void gst_rknn_face_mesh_set_property(GObject* object, guint id,
                                            const GValue* value, GParamSpec* spec) {
  auto* self = GST_RKNN_FACE_MESH(object);
  switch (id) {
    case PROP_MODEL_DIR:
      g_free(self->model_dir);
      self->model_dir = g_value_dup_string(value);
      break;
    case PROP_DRAW:
      self->draw = g_value_get_boolean(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, spec);
  }
}

static void gst_rknn_face_mesh_get_property(GObject* object, guint id,
                                            GValue* value, GParamSpec* spec) {
  auto* self = GST_RKNN_FACE_MESH(object);
  switch (id) {
    case PROP_MODEL_DIR:
      g_value_set_string(value, self->model_dir);
      break;
    case PROP_DRAW:
      g_value_set_boolean(value, self->draw);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, spec);
  }
}

static gboolean gst_rknn_face_mesh_start(GstBaseTransform* transform) {
  auto* self = GST_RKNN_FACE_MESH(transform);
  try {
    self->tracker = new rknn_face::Tracker(self->model_dir);
    return TRUE;
  } catch (const std::exception& e) {
    GST_ELEMENT_ERROR(self, RESOURCE, OPEN_READ, ("failed to load RKNN models"), ("%s", e.what()));
    return FALSE;
  }
}

static gboolean gst_rknn_face_mesh_stop(GstBaseTransform* transform) {
  auto* self = GST_RKNN_FACE_MESH(transform);
  delete self->tracker;
  self->tracker = nullptr;
  return TRUE;
}

static GstFlowReturn gst_rknn_face_mesh_transform_frame_ip(GstVideoFilter* filter,
                                                           GstVideoFrame* frame) {
  auto* self = GST_RKNN_FACE_MESH(filter);
  int width = GST_VIDEO_FRAME_WIDTH(frame);
  int height = GST_VIDEO_FRAME_HEIGHT(frame);
  int stride = GST_VIDEO_FRAME_PLANE_STRIDE(frame, 0);
  auto* pixels = static_cast<unsigned char*>(GST_VIDEO_FRAME_PLANE_DATA(frame, 0));
  if (stride < width * 3) {
    GST_ELEMENT_ERROR(self, STREAM, FORMAT, ("invalid RGB stride"), ("stride=%d width=%d", stride, width));
    return GST_FLOW_ERROR;
  }

  rknn_face::Image image{width, height, std::vector<unsigned char>(static_cast<size_t>(width) * height * 3)};
  for (int y = 0; y < height; ++y)
    std::memcpy(image.rgb.data() + static_cast<size_t>(y) * width * 3, pixels + y * stride, width * 3);

  rknn_face::Result result;
  bool found = false;
  try {
    found = self->tracker->process(image, result);
  } catch (const std::exception& e) {
    GST_ELEMENT_ERROR(self, STREAM, FAILED, ("RKNN inference failed"), ("%s", e.what()));
    return GST_FLOW_ERROR;
  }

  auto* meta = gst_buffer_add_rknn_face_meta(frame->buffer);
  meta->found = found;
  meta->score = result.score;
  meta->npu_ms = result.detector_ms + result.landmark_ms + result.blendshape_ms;
  if (found) {
    meta->landmark_count = static_cast<guint>(result.landmarks.size());
    meta->bbox[0] = result.bbox_x;
    meta->bbox[1] = result.bbox_y;
    meta->bbox[2] = result.bbox_w;
    meta->bbox[3] = result.bbox_h;
    meta->tongue_out = result.tongue_out;
    std::memcpy(meta->head_pose_deg, result.head_pose_deg.data(), sizeof(meta->head_pose_deg));
    std::memcpy(meta->blendshapes, result.blendshapes.data(), sizeof(meta->blendshapes));
    for (size_t i = 0; i < result.landmarks.size(); ++i) {
      meta->landmarks[i][0] = result.landmarks[i].x;
      meta->landmarks[i][1] = result.landmarks[i].y;
      meta->landmarks[i][2] = result.landmarks[i].z;
    }
    if (self->draw) {
      rknn_face::draw(image, result);
      for (int y = 0; y < height; ++y)
        std::memcpy(pixels + y * stride, image.rgb.data() + static_cast<size_t>(y) * width * 3, width * 3);
    }
  }
  return GST_FLOW_OK;
}

static void gst_rknn_face_mesh_finalize(GObject* object) {
  auto* self = GST_RKNN_FACE_MESH(object);
  delete self->tracker;
  g_free(self->model_dir);
  G_OBJECT_CLASS(gst_rknn_face_mesh_parent_class)->finalize(object);
}

static void gst_rknn_face_mesh_class_init(GstRknnFaceMeshClass* klass) {
  auto* object_class = G_OBJECT_CLASS(klass);
  auto* element_class = GST_ELEMENT_CLASS(klass);
  auto* transform_class = GST_BASE_TRANSFORM_CLASS(klass);
  auto* video_class = GST_VIDEO_FILTER_CLASS(klass);
  object_class->set_property = gst_rknn_face_mesh_set_property;
  object_class->get_property = gst_rknn_face_mesh_get_property;
  object_class->finalize = gst_rknn_face_mesh_finalize;
  g_object_class_install_property(object_class, PROP_MODEL_DIR,
      g_param_spec_string("model-dir", "Model directory", "Directory containing the three RKNN models",
                          "/app/models", static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(object_class, PROP_DRAW,
      g_param_spec_boolean("draw", "Draw landmarks", "Draw the face box and 478 landmarks on output frames",
                           TRUE, static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  gst_element_class_set_static_metadata(element_class, "RKNN face mesh", "Filter/Video",
      "RK3588 face detection, 478 landmarks, head pose and blendshapes", "CmST0us");
  gst_element_class_add_static_pad_template(element_class, &sink_template);
  gst_element_class_add_static_pad_template(element_class, &src_template);
  transform_class->start = gst_rknn_face_mesh_start;
  transform_class->stop = gst_rknn_face_mesh_stop;
  video_class->transform_frame_ip = gst_rknn_face_mesh_transform_frame_ip;
}

static void gst_rknn_face_mesh_init(GstRknnFaceMesh* self) {
  self->model_dir = g_strdup("/app/models");
  self->draw = TRUE;
  self->tracker = nullptr;
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
}

static gboolean plugin_init(GstPlugin* plugin) {
  return gst_element_register(plugin, "rknnfacemesh", GST_RANK_NONE, GST_TYPE_RKNN_FACE_MESH);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, rknnface,
                  "RKNN face mesh filter", plugin_init, "1.0.0", "LGPL",
                  "rknn-face-points", "https://github.com/CmST0us/rknn-face-points")
